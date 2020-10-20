import os
import re
import config
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm



# # # # # # #  PRE-CLUSTER   # # # # # # # 


config.data_dir = 'data_raw/'
config.cache_dir = 'cache/'
config.save_dir = 'listings_clean/'



def id_corp_groups(listings):
    print(' ', '—'*30)
    decoder = pd.read_excel(f'{config.data_dir}host decoder.xlsx') # custom host alias decoder
    decoder.drop(columns=['All Boston'], inplace=True)
    decoder.columns = decoder.loc[0].copy()
    decoder = decoder.loc[1:len(decoder)-2]
    decoder['Corp'] = decoder['Corp'].fillna(method='ffill')
    decoder = decoder[['Corp', 'host_id', 'host_name']]
    found = [h_id for h_id in decoder.host_id.unique() if h_id in list(listings.host_id)]
    print('  > Searching listings data for alias host accounts...')

    
    # compile host corp group map
    host_group = {}
    group_idx = 0
    for corp in decoder.Corp.unique():
        corp_data = decoder[decoder.Corp==corp]
        hosts = list(corp_data.host_id.unique())
        host_names = list(corp_data.host_name.unique())
        if len(hosts)>0:
            group_idx +=1
            corp = re.sub('[A-Z]- ?', '', corp)
            h_idx = f'CorpGroup{group_idx}'
            for host in hosts:
                host_group[host] = {'host_name': corp,
                                    'host_id': h_idx,
                                    'group_hosts': ', '.join(host_names),
                                   }
    print(f'    > Found {len(found)}/{decoder.host_id.nunique()} active alias host accounts...')

    # isolate listings data
    for row in listings.index:
        row_data = listings.loc[row]
        if row_data.host_id in host_group.keys():
            listings.loc[row, 'host_alias'] = listings.loc[row, 'host_name']
            listings.loc[row, 'alias_id'] = listings.loc[row, 'host_id']
            listings.loc[row, 'host_name'] = host_group[row_data.host_id]['host_name']
            
            listings.loc[row, 'host_name'] = host_group[row_data.host_id]['host_name']
            listings.loc[row, 'host_id'] = host_group[row_data.host_id]['host_id']
    
    print(f"  >>> Generated {group_idx} new CorpGroup host IDs")
    return listings


def load_parsed_listings(file='listings.csv'):
    print('—'*60)
    print(f"Loading & parsing listings data")
    print('—'*60)
    listings = pd.read_csv(f'{config.data_dir}{file}')
    print(f"  > '{file}' loaded")
    # check global config
    listings = id_corp_groups(listings)
    
    
    # Parse & verify license numbers...
    print(f"  > Parsing license numbers")
    
    listings.price = listings.price.apply(lambda x: float(''.join(x.replace(',', '').split('$')[1])))
    def parse_str_num(l): # from listings data
        license = str(l).lower()
        if 'hospital' in license:
            return 'STRHS'
        if 'business' in license:
            return 'STRES'
        if 'hotel' in license:
            return 'STRLH/STRBB'
        if 'str' in license:
            return ''.join(re.findall('\d{6}', license))
        elif 'C0' in license:
            return license.upper().split('\n')[0]
       
        # probably either a number or missing:
        if license!='nan': 
            try: return ''.join(re.findall('\d{6}', license))
            except: pass
            
        else: return 'Missing' # none claimed
        
    listings.license = listings.license.apply(parse_str_num)
    print(f"    > Imported + licenses parsed")
    
    print(' ', '—'*30)
    print("  > Importing ISD registry data...")
    
    # load ISD data
    def parse_license_num(l):
        return (''.join(re.findall('[\d]*', str(l).split('\n')[0]))) or 0
    # get license status
    isd = pd.read_csv('data_raw/ISD data.csv')
    isd['license_num'] = isd['License #'].apply(parse_license_num).astype(str)
    print("    > Imported + licenses parsed")
    
    
    
    # most recent first, to index with .values[0] in verification
    isd['issued'] = pd.to_datetime(isd['Issued Date'].fillna('2019-01-01'))
    isd.sort_values('issued', ascending=False)
    
    license_cat_dict = {
    'HS':{'name':'Home Share Unit',
          'maxbeds':5,
          'maxguests':10},
    'LS':{'name':'Limited Share Unit',
          'maxbeds':3,
          'maxguests':6},
    'OA':{'name':'Owner-Adjacent Unit',
          'maxbeds':5,
          'maxguests':10},
    'STRES':{'name':'Exempt: Executive Suite',
          'maxbeds':9999,
          'maxguests':999},
    'STRLH':{'name':'Exempt: Lodging House',
          'maxbeds':999,
          'maxguests':999},
    'STRBB':{'name':'Exempt: Bed & Breakfast Suite',
          'maxbeds':999,
          'maxguests':999},
    'STRLH/STRBB':{'name':'Exempt: Lodging House / B&B'},
    'STRHS':{'name':'Exempt: Hospital Stays',
          'maxbeds':999,
          'maxguests':999},
    }
    
    # LONG LOOP ——————————————...
    # get given license status...
    config.isd = isd
    print(' ', '—'*30)
    print('  > Parsing license statuses...')
    for license in tqdm(listings.license.unique()):
        
        listings_license_data = listings[listings.license==license].copy()
        full_status, simple_status = False, False
        # NO LICENSE # CLAIMED, JUST EXEMPT OR MISSING
        if 'STR' in license or 'Missing' in license:
            ISD_address = np.nan
            ISD_status = np.nan
            count_listings = np.nan
            license_cat_max_exceed = np.nan
            if 'STR' in license:
                ISD_category = license_cat_dict[license]['name'].replace('(verified)', '(not verified)')
                status = simple_status = "Exempt (not verified)"
                full_status = f"Exempt (not verified): {ISD_category.split(': ')[1]}"
            elif 'Missing' in license:
                ISD_category = np.nan
                status = 'No license claimed' 

            
        else:   # LICENSE FOUND!   
            count_listings = len(listings_license_data)
            
            if license in isd['license_num'].values:  # license found in ISD data:
                isd_license_data = isd[isd.license_num==license]
                
                found_status = isd_license_data['Status'].values[0]
                address = isd_license_data['Address'].values[0]
                cat = isd_license_data['Category'].values[0]
                category_dict = license_cat_dict[cat]
                ISD_category = category_dict['name']
        
# CHECK FOR EXCEEDED LIMIT OF LISTINGS OR ACCODMODATIONS:
                
                # #   ""  Occupancy shall be limited to five bedrooms or ten   ""
                # #   ""  guests in a Home Share Unit, whichever is less.      "" from ordinance
            
                count_accomms = listings_license_data.accommodates.sum()
                license_cat_max_exceed = np.nan
                if count_listings <= count_accomms:
                    # do beds(≈n_listings) exceed category limit?
                    if count_listings > category_dict['maxbeds']:
                        license_cat_max_exceed = 'listing'
                elif count_accomms <= count_listings: 
                    # do guests(≈accomdations) exceed category limit?
                    if count_accomms > category_dict['maxguests']:
                        license_cat_max_exceed = 'guest'
                
                # get given license status
                
                if 'Active' in found_status:   
                    ISD_status = found_status
                    ISD_address = address
                    
                    if 'Exempt' in ISD_category:
                        status = simple_status = "Exempt (verified)"
                        full_status = f"{status}: {ISD_category.split(': ')[1]}"
                    
                    elif str(license_cat_max_exceed)!='nan':
                        status = simple_status = full_status = 'Active (limit exceeded)'
                    else:
                        status = 'Active'
                else:
                    status = 'Expired/Void/Revoked' # group these for simple version
                    full_status = found_status
                    ISD_status = found_status
                    ISD_address = np.nan
                    
            
            # license not found in ISD data
            else: 
                status = 'Not found (fabricated)' 
                ISD_status = status
                ISD_address = address
                license_cat_max_exceed = np.nan
                
    # ----- v still iterating through this licenses! v ---- assign license details to listings
        
        listings.loc[listings_license_data.index, 'status'] = status
        listings.loc[listings_license_data.index, 'ISD_status'] = ISD_status
        listings.loc[listings_license_data.index, 'ISD_address'] = ISD_address
        listings.loc[listings_license_data.index, 'ISD_category'] = ISD_category
        listings.loc[listings_license_data.index, 'license_listing_count'] = count_listings
        listings.loc[listings_license_data.index, 'license_cat_max_exceed'] = license_cat_max_exceed


        if simple_status:
            listings.loc[listings_license_data.index, 'simple_status'] = simple_status
        else: simple_status = status
        if full_status:
            listings.loc[listings_license_data.index, 'full_status'] = full_status
            
        listings.loc[listings_license_data.index, 'simple_status'] = simple_status

        
    #——————— END LOOP
    listings.full_status = listings.full_status.fillna(listings.simple_status)

                
    print(f"    > Matched {listings.ISD_address.notna().sum()} to ISD data")

    
    pct_active = round(listings.status.value_counts(normalize=True)['Active']*100)
    print(f"  >>> {int(pct_active)}% of total listings have an Active license")
    print(' ', '—'*30)
    
    print("  > Searching for license details of listings in ISD data...")  

    # check if license number is used by other hosts
    def find_shared_licenses(listings):
        for host in tqdm(set(listings.host_id)): # isolate data for each host 
            host_data = listings[listings.host_id==host]
            # iterating host-license groups allows exclusion of the self-host from list of others
            for license in host_data.license.unique(): # isolate data for each license by host
                group_idx = host_data[host_data.license==license].index
                if 'STR' not in license and 'Missing' not in license:
                    all_hosts = set(listings[listings.license==license].host_id.unique())
                    if len(all_hosts)>1: # more than one host using this license
                        all_hosts.remove(host)
                        listings.loc[group_idx, 'other_hosts'] = ', '.join([str(h) for h in all_hosts])     
                else: listings.loc[group_idx, 'other_hosts'] = np.nan
        return listings
    print('  > Defining licenses used by multiple hosts ')
    
    listings = find_shared_licenses(listings)
    
    
    # Make shared index between dataframes:
    def get_isd_index(listings):
        print('Mapping exact index back to listings dataframe...') # not currently needed / used, but interesting
        isd_match_idx = {}
        for license in listings.license.unique(): # save time   
            list_idx = listings[listings.license==license].index
            if 'Exempt' in license or 'Missing' in license: continue
            matches = ''
            for isd_idx in isd.index:
                if license in isd.loc[isd_idx, 'license_num']:
                    matches+=str(isd_idx)+' '

            listings.loc[list_idx, 'isd_index'] = matches
        
    listings = listings.rename(columns={'host_name':'host'})
    print('—'*60)
    return listings[['id', 'listing_url', 'name', 'host_id', 'host_url',
       'host', 'neighbourhood', 'neighbourhood_cleansed',
       'neighbourhood_group_cleansed', 'latitude', 'longitude',
       'property_type', 'room_type', 'accommodates', 'bedrooms',
       'beds', 'amenities', 'price', 'minimum_nights', 'maximum_nights',
       'minimum_minimum_nights', 'has_availability',
       'availability_30', 'availability_60', 'availability_90',
       'availability_365', 'number_of_reviews', 'license', 'instant_bookable',
       'host_alias', 'alias_id', 'status', 'ISD_status', 'ISD_address',
       'ISD_category', 'license_listing_count', 'license_cat_max_exceed', 'simple_status',
       'full_status', 'other_hosts']] 