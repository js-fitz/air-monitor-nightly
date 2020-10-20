import os
import re
import time
import pickle
import config
import numpy as np
import pandas as pd
from sklearn.cluster import dbscan
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="3joemail@gmail.com")
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


# config to host alias decoder mode
def decoder_mode(active):
    print('—'*60)
    if active:
        config.host_decoder = True
        config.cache_dir = 'cache/corp_grouped/'
        config.save_dir = 'data_cleaned/corp_grouped/'
        print('> Corporation group decoder mode ACTIVATED')
    else:
        config.host_decoder = False
        config.cache_dir = 'cache/host/'
        config.save_dir = 'data_cleaned/host/'
        print('> Corporation group decoder mode is OFF')
        
    print('  > Cache directory set to', config.cache_dir)
        
# drops hosts with fewer than 3 listings
def drop_little_guys(listings, min_listings=3): 
    print(' ', '—'*30)
    print(f"> Dropping hosts with fewer than {min_listings} listings...")
    vcs = listings.host_id.value_counts()
    print(f"  > Before: {len(vcs)} hosts | {len(listings)} listings")
    big_league = vcs[vcs>=min_listings].index
    for row in listings.index:
        if listings.loc[row, 'host_id'] not in big_league:
            listings.drop(row, inplace=True)
    print(f">>> Remaining: {len(big_league)} hosts | {len(listings)} listings")
    return listings


def id_corp_groups(listings):
    print(' ', '—'*30)
    print('> Loading corporation group decoder data...')
    decoder = pd.read_excel('data_raw/host decoder.xlsx') # custom host alias decoder
    decoder.drop(columns=['All Boston'], inplace=True)
    decoder.columns = decoder.loc[0].copy()
    decoder = decoder.loc[1:len(decoder)-2]
    decoder['Corp'] = decoder['Corp'].fillna(method='ffill')
    print(f'  > {decoder.Corp.nunique()} corp groups using {decoder.host_id.nunique()} aliases')
    decoder = decoder[['Corp', 'host_id', 'host_name']]
    found = [h_id for h_id in decoder.host_id.unique() if h_id in list(listings.host_id)]
    print('> Searching listings data for alias host accounts...')

    
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
    print(f'  > Found {len(found)}/{decoder.host_id.nunique()} active alias host accounts')

    # isolate listings data
    for row in listings.index:
        row_data = listings.loc[row]
        if row_data.host_id in host_group.keys():
            listings.loc[row, 'host_alias'] = listings.loc[row, 'host_name']
            listings.loc[row, 'alias_id'] = listings.loc[row, 'host_id']
            listings.loc[row, 'host_name'] = host_group[row_data.host_id]['host_name']
            
            listings.loc[row, 'host_name'] = host_group[row_data.host_id]['host_name']
            listings.loc[row, 'host_id'] = host_group[row_data.host_id]['host_id']

    
    print(f'  > {listings.host_alias.notna().sum()} listings')
    print(f'  > {listings.alias_id.nunique()} alias accounts')
    print(f"  > Generated {group_idx} new corporate group host IDs for alias accounts")
    return listings


def load_listings(file='data_raw/listings.csv'):
    print('—'*60)
    print(f"> Loading listings data", end='')
    listings = pd.read_csv(file)
    print(f"  > '{file}' loaded")
    # check global config
    if config.host_decoder: listings = id_corp_groups(listings)
    
    
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
        
    print(' ', '—'*30)
    listings.license = listings.license.apply(parse_str_num)
    print(f"  > Imported + licenses parsed")
    
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
        listings.loc[listings_license_data.index, 'license_f'] = count_listings
        listings.loc[listings_license_data.index, 'license_cat_max_exceed'] = license_cat_max_exceed


        if simple_status:
            listings.loc[listings_license_data.index, 'simple_status'] = simple_status
        else: simple_status = status
        if full_status:
            listings.loc[listings_license_data.index, 'full_status'] = full_status
            
        listings.loc[listings_license_data.index, 'simple_status'] = simple_status

        
    #——————— END LOOP
    listings.full_status = listings.full_status.fillna(listings.simple_status)

                
    print(f"  > Matched {listings.ISD_address.notna().sum()} to ISD data")

    
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
        print('—'*60)

    return listings  

    # listings = get_isd_index(listings)


# converts DBSCAN epsilon value from feet to lat/long (approximately) 
def eps_calc(desired_ft):
    # (epsilon will shift slightly from geodesic distances)
    cal = [ (42.354304, -71.069223),  # 500 ft calibration
            (42.353843, -71.070958) ]
    
    x_diff = (cal[0][0] - cal[1][0])**2
    y_diff = (cal[0][1] - cal[1][1])**2
    ft_factor = np.sqrt(x_diff + y_diff) / 500 # calibration

    epsilon = round(desired_ft*ft_factor, 8) 
    return epsilon


def cluster_listings(listings, round_1_params): # e=.002 (in lat/lng) => approx 500 ft
          
    print('\n', '—'*60)
    print('> Clustering PHASE 1... (listings ——> sub clusters)')
        
    listings.reset_index(inplace=True, drop=True)
    
    # host-license group index to iterate
    group_id = 1 # start at one so all integer

    # for logging:
    total_count = 0
    single_listing = 0
    pre_clustered = 0
    exploded_n = 0 # expanded
    created_n = 0 # expansions
    
    print(' ', '—'*30)
    print(f'> Running DBSCAN on {len(listings)} listings ({len(set(listings.host_id))} host_ids)...')
    for host in tqdm(set(listings.host_id)): # isolate data for each host 
        host_data = listings[listings.host_id==host]
        
        # iterating host-license groups
        for license in host_data.license.unique(): # isolate data for each license by host
            total_count += 1
            
            # assign host-license group_id index
            
            #if fname = 'cluster_phase1.cache'
            #if fname in os.listdir(config.cache_dir):
            #    with open(config.cache_dir+fname,'rb') as file:
            #        centroid_data = pickle.load(file)
            ## to compile dicts of info about each centroid
            #else: centroid_data = {}
            #print(f'  > Found {len(centroid_data)} cached geo stats')
            #print(' ', '—'*30)
            
            group_idx = listings[(listings.license==license) & (listings.host_id==host)].index
            listings.loc[group_idx, 'license_group'] = group_id
            group_id+=1
         

            # if single-listing group, single cluster & finish
            if len(group_idx)<2:
                single_listing += 1
                listings.loc[group_idx, 'geo_cluster'] = 0
                continue
                
            epsilon = eps_calc(round_1_params['epsilon'])
             
            # CAN CREATE A DYNAMIC EPSILON HERE BASED ON LICENSE GROUP COUNT

            # if more than one listing, DBSCAN within this host-license group for geo-clusters
            group_data = listings.loc[group_idx]
            classes = dbscan(
                group_data[['latitude', 'longitude']].values,
                eps = epsilon,
                min_samples = round_1_params['min_samples'],
                metric='minkowski',
                algorithm='auto',
                leaf_size= round_1_params['leaf_size'],
                p=2,)[1]
            
            # if multiple sub-clusters detected, save DBSCAN results 
            n_classes = len(set(classes))
            if n_classes>1:
                exploded_n += 1
                created_n += n_classes
                listings.loc[group_idx, 'geo_cluster'] = classes
            else:
                # 0 = already-clustered
                listings.loc[group_idx, 'geo_cluster'] = 0 
                pre_clustered += 1
    

    print(' ', '—'*30)
    print('Searching for clusters within HOST-LICENSE groups...')
    # create sub-group group ids
    for row in listings.index:
        lg = str(int(listings.loc[row, 'license_group']+1))
        sc = str(int(listings.loc[row, 'geo_cluster']+1))
        listings.loc[row, 'sub_group'] = lg+'.'+sc
        

    print(f' > {total_count} existing groups found')
   
    # count duplicates
    duplicators = listings[listings.other_hosts.notna()]
    if len(duplicators)>0:
        print(' > %s licenses are SHARED by mutiple hosts (%s accounts)'%(
            duplicators.license.nunique(),
            duplicators.host_id.nunique(),
        ))
    print(' ', '—'*30)
    print('> Compiling initial sub-clusters...')
    print(f'  + < {single_listing} host-license groups have only one listing ')
    print(f'    > {exploded_n+pre_clustered} license groups have multiple')
    print(f'  +   < {pre_clustered} groups exhibit only 1 cluster')
    print(f'      > {exploded_n} groups exhibit multiple clusters...')
    print(f'  +   < {created_n} geo-clusters generated with DBSCAN ')
    print(f'>>> {listings.sub_group.nunique()} clusters identified')
    print('—'*60)
    
    return listings
        


# generates a df with subgroup centroids as rows and group info + geo stats as columns
def find_subgroups(listings):
    print('—'*60)
    print(f'> Calculating centroid geo stats for {len(set(listings.sub_group))} sub groups...')
    
    # for subgroups, before re-clustering
    fname = 'cluster_phase1.cache'
    if fname in os.listdir(config.cache_dir):
        with open(config.cache_dir+fname,'rb') as file:
            centroid_data = pickle.load(file)
    # to compile dicts of info about each centroid
    else: centroid_data = {}
    print(f'  > Found {len(centroid_data)} cached geo stats')
    print(' ', '—'*30)
    
    cache_found = 0
    new_search = 0
    
    for c_group in tqdm(listings.sub_group.unique()): # iterate through centroid groups
        c_data = listings[listings.sub_group==c_group]
        
    # basic info about centroid group
        # stats needing all listings
        count_listings = len(c_data)
        avg_USD = round(c_data['price'].mean(), 2)
        
        # stats that are universal for all listings: reference only one row...
        c_data = listings[listings.sub_group==c_group].reset_index().loc[0]

        # skip if same file exists (verify by listing id series)
        if c_group in centroid_data.keys():
            existing_ids = centroid_data[c_group]['host_id']
            new_ids = c_data['host_id']
            new_listings = centroid_data[c_group]['count_listings']
            if existing_ids == new_ids and count_listings==new_listings:
                cache_found += 1
                centroid = centroid_data[c_group]
                continue
        else: new_search +=1
        centroid = {'host': c_data['host_name'],
                    'host_id': c_data['host_id'],
                    'license': c_data['license'],
                    'license_cat_max_exceed': c_data['license_cat_max_exceed'],
                    'license_status': c_data['status'],
                    'ISD_address': c_data['ISD_address'],
                    'license_group': c_data['license_group'],
                    'geo_cluster': c_data['geo_cluster'],
                    'sub_group': c_data['sub_group'],
                    'other_hosts': c_data['other_hosts'],
                    'count_listings': count_listings,
                    'avg_USD/night': avg_USD, }
        
    # centroid group stats for multi-listing groups
        # stats need all listings again
        c_data = listings[listings.sub_group==c_group]
        if len(c_data)>1:
            geo_stats = c_data.describe()
            for ll in ['latitude', 'longitude']:
                for stat in geo_stats.index[1:]:
                    if '%' not in stat:
                        centroid[f'{stat}_{ll}'] = geo_stats.loc[stat, ll]
     
      # manually insert base stats for single-listing groups
        else:
            for ll in ['latitude', 'longitude']:
                for stat in ['min', 'mean', 'max']:
                    centroid[f'{stat}_{ll}'] = c_data[ll].values[0]
                          
        centroid_data[c_group] = centroid
      
    
    if new_search:
        with open(config.cache_dir+fname,'wb') as file:
            pickle.dump(centroid_data, file)
            
    print(f'  > {cache_found} retrieved from the cache')
    print(f'> {new_search} new locations geocoded & added to cache')
    print('—'*60)
    return pd.DataFrame(list(centroid_data.values())).sort_values('count_listings', ascending=False)





# checks to see if any clusters by a host should be combined
def cluster_clusters(centroids, listings, round_2_params, phase, phases):

    
    if phase==0 or phases==2:
        old_c_feat='sub_group'
    else: # more than 2 phases:
        old_c_feat = f'sub_group_layer_{phase}'
    
    if phase==phases: # no phases left
        new_c_feat = 'centroid_group'
    else: # multi left
        new_c_feat = f'sub_group_layer_{phase+1}'
    
    print('\n', '—'*60, '\n', '—'*60)
    print(f"> Clustering PHASE 2-{phase*'I'}... ['{old_c_feat}'] ——> ['{new_c_feat})']")
    sub_groups = 0
    super_groups = 0
    centroids = centroids.reset_index(drop='True')
    print(' ', '—'*30)

    
    print(f'> Merging detected clusters (within {centroids.host_id.nunique()} HOST groups)...')
    print(f'> {centroids[old_c_feat].nunique()} clusters detected')
    
    ax = config.fig.add_subplot(int(f'1{2*(phases+1)}{(phase+1)}'))
    ax.hist(listings[old_c_feat].values, bins=15, color='orange')
    ax.set_title(f"['{old_c_feat}'] size distribution:")
    ax.set_xticks(range(0, 150, 10))
    ax.set_xticklabels(range(0, 150, 10))

    ax.tick_params(axis='x', labelrotation=90)

    for host in tqdm(centroids.host_id.unique()):
        host_centroids_idx = centroids[centroids.host_id==host].index
        host_centroids = centroids.loc[host_centroids_idx]
        
        classes = dbscan(
            host_centroids[['mean_latitude', 'mean_longitude']].values,
            eps = eps_calc(round_2_params['epsilon']),
            min_samples = round_2_params['min_samples'],
            metric='minkowski',
            algorithm='auto',
            leaf_size = round_2_params['leaf_size'],
            p=2,)[1]
        
        if len(classes)!=len(host_centroids_idx):
            print('host:', host, '- phase 1 centroids:', len(host_centroids_idx))
            print('> found ', len(classes), ' sub-classes in phase 2\n')
        
        centroids.loc[host_centroids_idx, 'cluster'] = classes
      
        if len(set(classes)) < host_centroids[old_c_feat].nunique(): # bool clustered?
            sub_groups += 1
        super_groups += len(set(classes)) # for logg
        
    print(f'> {sub_groups} sub-clusters combined --> {super_groups} (super) clusters groups remain')
    

    # create final centroid ids
    
    # LOAD & APPLY STATIC HOST NAMES (avoid deleting this file...)
    print(f'—'*30)

    print(f'  > Acquring static host_num index...')
    fname = 'host_name_index.cache'
    if fname in os.listdir(config.cache_dir):
        with open(config.cache_dir+fname,'rb') as file:
            host_name_index = pickle.load(file)
        print(' ', '—'*30)
    else: host_name_index = {}
    last_max = len(host_name_index)
    print(f'  > Found {last_max} cached host names')
    new_search = 0
    found = 0
    for i, host_id in enumerate( # sort for hosts with highest listing counts
    listings.host_id.value_counts(ascending=False).index):
        host_data_idx = centroids[centroids.host_id==host_id].index
        if host_id not in host_name_index.keys():
            host_name_index[host_id] = last_max + i # keep stacking on the old counter <———————
            centroids.loc[host_data_idx, 'host_num'] = i
            new_search += 1
        else:
            centroids.loc[host_data_idx, 'host_num'] = host_name_index[host_id]
            found +=1
    if new_search:
        with open(config.cache_dir+fname,'wb') as file:
            pickle.dump(host_name_index, file)
            print(f'  > Added {new_search} new static host name indexes to cache')
    if found: print(f'  > Found {found} static host name indexes in the cache')
    print(f'  > Cache saved ({len(host_name_index)} total items) > {fname}')
    print(f'—'*30)

    
    # TREAT [host_num] as the new HOST ID for the next section to access static centroid cache
    
    # CREATE OR USE CACHED CENTROID IDS — AND SAVE BIG SUB-CLUSTERS AS SUPER-CLUSTERS
    print('> Mapping new centroid group values onto listings data...')
    super_counter = {h: 1 for h in centroids.host_num.unique()} # to count superclusters within a host
    parent_cluster_dict = {} # to map back to listings
    super_groups = 0
    norm_groups = 0
    # iterate merged clusters found by PHASE 2
    for big_cluster in centroids.cluster.unique():        
        
        bc_data = centroids[centroids.cluster==big_cluster]
        
        host_num = bc_data.host_num.values[0]
        
        # iterate split clusters found by previous phase (within each merged cluster)
        for geo_cluster in bc_data[old_c_feat].unique():
            gc_data = centroids[centroids[old_c_feat]==geo_cluster]
            gc_listing_ids = listings[listings[old_c_feat]==geo_cluster]['id'].values
            
            # check for 90% here against cached version, use same cluster number and end loop if close match
            # ^^ ACTUALLY DO THIS^ IN A SEPARATE ITERATION ^^
            
            # if big sub-cluster, save it as a "super-group" (IGNORE PHASE 2 II results)
            if gc_data.count_listings.values[0]>=round_2_params['min_super_save']: 
                sub_inferred = f'{int(super_counter[host_num])}s'
                super_counter[host_num] += 1 # step up super cluster counter if new find
                super_groups += 1 # log
                
            else: # ACCEPT PHASE 2 results - dbscan big-scan class
                sub_inferred = f'{int(big_cluster+1)}'
                norm_groups +=1  # log 
            
            centroid_group = f'{int(host_num)}-{sub_inferred}'
            centroids.loc[gc_data.index, new_c_feat] = centroid_group
            parent_cluster_dict[geo_cluster] = centroid_group # for mapping back to listings
        
    # map final centroid groups back onto listings
    print(f'  > Mapping final centroid group index back to listings')
    listings[new_c_feat] = listings[old_c_feat].map(parent_cluster_dict)
    for c in ['license_group', 'geo_cluster', old_c_feat, 'host_num']:
        if c in centroids.columns:
            centroids = centroids.drop(columns=[c])
        if c in listings.columns:
            listings = listings.drop(columns=[c])
    print(f"> Cluster PHASE {(phase+1)*'I'} COMPLETED ['{old_c_feat}'] ——> ['{new_c_feat})']")
    print(f'  > Kept {super_groups} super (sub) groups separate')
    print(f'  > Grouped {norm_groups} smaller clusters')
    print(f'>>> {centroids[new_c_feat].nunique()} total clusters remain')
    

    ax = config.fig.add_subplot(int(f'1{2*(phases+1)}{(phase+2)}'))
    ax.hist(listings[new_c_feat].values, bins=20)
    ax.set_title(f"['{new_c_feat}'] size distribution:")
    ax.set_xticks(range(0, 150, 10))
    ax.set_xticklabels(range(0, 150, 10))
    ax.tick_params(axis='x', labelrotation=90)

    print('—'*60)  
    try: assert(listings[new_c_feat].isna().sum()==0)
    except: raise BaseException(
            "ERROR: Some listings were not assigned centroid groups.\nTry clearing cache & try again!")
    try: assert( set(listings[new_c_feat].values) == set(centroids[new_c_feat].values) )
    except: 
        message = f"{new_c_feat} uniques do not match between centroids and listings data.\nTry clearing cache & try again!"
        print(f'WARNING: {message}')
        #raise BaseException(f"ERROR: {message}")
            
    return centroids, listings



# generates a df with subgroup centroids as rows and group info + geo stats as columns
def merge_clusters(centroids, listings):
    print('—'*60)
    new_search = 0
    cache_found = 0
          
    print(f'> Calculating centroid geo stats for {len(set(centroids.centroid_group))} clusters...')
    
    # for subgroups, before re-clustering
    fname = 'cluster_phase2.cache'
    if fname in os.listdir(config.cache_dir):
        with open(config.cache_dir+fname,'rb') as file:
            centroid_data = pickle.load(file)
        print(f'  > Found {len(centroid_data)} cached geo stats.')
        print(' ', '—'*30)
    else: centroid_data = {}

    # compile dicts of info about each centroid
    
    for c_group in tqdm(centroids.centroid_group.unique()): # iterate through centroid groups
        c_data = centroids[centroids.centroid_group==c_group]
        l_data = listings[listings.centroid_group==c_group]
        
    # basic info about centroid group
        
        # skip if same file exists (verify number of listings per license in group)
        if c_group in centroid_data.keys():
            existing_count = centroid_data[c_group]['license_dict']
            new_count = dict(l_data.license.value_counts())
            if existing_count == new_count:
                cache_found += 1
                continue
        else: new_search +=1
        
        count_listings = len(l_data)
        license_dict = dict(l_data.license.value_counts())
        avg_USD = round(l_data['price'].mean(), 2)        
        status_dict = dict(l_data.status.value_counts())
        alias_dict = dict(l_data.alias_id.astype(str).replace('.0', '').value_counts())
        ISD_address_dict = dict(l_data.ISD_address.value_counts())
        if l_data.license_cat_max_exceed.notna().sum()>0:
            license_exceed_type = l_data.license_cat_max_exceed.value_counts().index[0]
        else:
            license_exceed_type = np.nan

        centroid = {'centroid_group': c_data['centroid_group'].values[0],
                    'count_listings': count_listings,
                    'count_licenses': c_data.license.nunique(),
                    'license_dict': license_dict,
                    'status_dict': status_dict,
                    'license_exceed_type': license_exceed_type,
                    'ISD_adress_dict':ISD_address_dict,
                    'host': c_data['host'].values[0],
                    'host_id': c_data['host_id'].values[0],
                    'alias_dict': alias_dict,
                    'shared_license': l_data.other_hosts.notna().sum() > 0,
                    'avg_USD/night': avg_USD,
                   }
        
    # centroid group stats for multi-listing groups
        
        coord_names = ['latitude', 'longitude']
        geo_statnames = ['mean', 'std', 'min', 'max'] 

        if len(c_data)>1:
            geo_stats = l_data[coord_names].describe()
            for ll in coord_names:
                for stat in geo_statnames:
                    centroid[f'{stat}_{ll}'] = geo_stats.loc[stat, ll]

      # manually insert base stats for single-listing groups
        else:
            for ll in coord_names:
                for stat in geo_statnames:
                    centroid[f'{stat}_{ll}'] = c_data[f'{stat}_{ll}'].values[0]
                    
        centroid_data[c_group] = centroid
        
            
    print(f'  > {cache_found} retrieved from the cache')
    print(f'  > {new_search} new locations geocoded & added to cache')
    print('—'*60)
    return pd.DataFrame(list(centroid_data.values())).sort_values('count_listings', ascending=False)






def geocode(centroids):
    print('—'*60)
    print('> Reverse geocoding cluster centroids with nominatim...')
    centroids = centroids.copy()
    
    fname = 'geocode.cache'
    if fname in os.listdir(config.cache_dir):
        with open(config.cache_dir+fname, 'rb') as file:
            geo_dict = pickle.load(file)
    else: geo_dict = {}
    print(f'  > Found {len(geo_dict)} pre-geocoded locations in cache')
    
    cache_found = 0
    new_find = 0
    for ridx in tqdm(centroids.index):
        
        if str(centroids.loc[ridx, 'mean_latitude'])=='nan': # just in case
            centroids.loc[ridx, 'address_gcoded'] = '?' 
            continue
            
        key = f"{centroids.loc[ridx,'mean_latitude']}, {centroids.loc[ridx,'mean_longitude']}"
            
        if key in geo_dict.keys():
             cache_found += 1            
        else:
            new_find += 1
            try:
                georeq = geolocator.reverse(key, addressdetails=True)
                address = georeq.raw['address']

            except:
                with open('cache'+fname,'wb') as file:
                    pickle.dump(geo_dict, file)
                raise BaseException('Connection lost — autosaved cache')
                
                
            geo_dict[key] = address
            time.sleep(.15)
        
        
        for component in geo_dict[key].keys():
            centroids.loc[ridx, f'GCODE_{component}'] = geo_dict[key][component]

    
    # log & save to cache
    print(f'  > {cache_found} retrieved from the cache')
    print(f'> {new_find} new locations geocoded & added to cache')
    with open(config.cache_dir+fname,'wb') as file:
        pickle.dump(geo_dict, file)
    
    print('—'*60)
    return centroids


def merged_radial_stats(centroids, listings):
    print('—'*60)
    print('> Analyzing centroid radial stats...')

    distances = []
    print('  > Calculating listing distances from associated centroids...')
    
    for l_idx in tqdm(listings.index): # iterate listings
        l_lat = listings.loc[l_idx, 'latitude']
        l_lng = listings.loc[l_idx, 'longitude']
        c_group = listings.loc[l_idx, 'centroid_group']
        c_listings = listings[listings.centroid_group==c_group]
        
        if len(c_listings)==1 or :
            continue # single-listing centroid
        else:
            c_idx = centroids[centroids.centroid_group==c_group].index

            c_lat = centroids.loc[c_idx, 'mean_latitude'].values[0]
            c_lng = centroids.loc[c_idx, 'mean_longitude'].values[0]
            
            distance = geodesic((c_lat,c_lng), (l_lat,l_lng)).feet
            listings.loc[l_idx, 'centroid_distance'] = distance
                
    
    # calculate location confidence (using avg. distance from centroid)
    print(' ', '—'*30)
    print('  > Calculating centroid location confidence...')
    distance_dict = {}
    distance_radius = {}
    for c_group in tqdm(centroids.centroid_group.unique()):  # iterate listings
        
        listings_data = listings[listings.centroid_group==c_group] 
        c_idx = centroids[centroids.centroid_group==c_group].index
        
        max_distance = round(listings_data['centroid_distance'].max(), 5)
        if len(listings_data)<3:  
            centroids.loc[c_idx, 'confidence'] = 'unknown (too few)'
        elif max_distance < 50:
            centroids.loc[c_idx, 'confidence'] = 'unknown (too close)'
        else:
            centroids.loc[c_idx, 'confidence'] = f'within {round((max_distance)/50)*50} ft.'
        
        centroids.loc[c_idx, 'max_listing_distance'] = max_distance
        centroids.loc[c_idx, 'avg_listing_distance'] = round(
            listings_data['centroid_distance'].mean(), 4)
        centroids.loc[c_idx, 'std_listing_distance'] = round(
            listings_data['centroid_distance'].std(), 4)

    print('—'*60)
    
    return centroids, listings

     
def run(host_decoder_mode, round_1_params, round_2_param_list, testing=False):
    print('PREPARING DATA')
    start = time.perf_counter()
    decoder_mode(host_decoder_mode)
    listings = load_listings()
    
    listings_all = listings.copy()
    
    
    # drop sub-3-listing hosts
    listings = drop_little_guys(listings) 
    
    listings = cluster_listings(listings, round_1_params)
    centroids = find_subgroups(listings)
    if testing: return centroids
    
    phases = len(round_2_param_list)-1
    for phase, round_2_params in enumerate(round_2_param_list):
        
        config.fig = plt.figure(figsize=(40, 2*(phases+1)))
        
        centroids, listings = cluster_clusters(centroids, listings, round_2_params, phase, phases)
        
    plt.show()
    
    centroids = merge_clusters(centroids, listings)
    centroids = geocode(centroids)
    centroids, listings = merged_radial_stats(centroids, listings)

    try: centroids.drop(columns=['cluster', 'host_num'], inplace=True)
    except: pass
    centroids.drop(columns=['std_latitude', 'min_latitude', 'max_latitude',
                            'std_longitude', 'min_longitude', 'max_longitude',
                           ], inplace=True)
    centroids['ISD_found'] = centroids.ISD_adress_dict.apply(lambda x: bool(len(x)>0))

    centroids = centroids.sort_values('count_listings', ascending=False)
    listings = listings.sort_values('id')
    
    

    
    print(f' > Adding {len(listings_all)} small listings back for home page... ')
    listings_all = pd.concat([listings, listings_all])
    listings_all.drop_duplicates(subset=['id'], inplace=True)
    listings_all.to_csv(config.save_dir+'listings_all.csv', index=False)
    
    print('—'*60)
    print('Saving files to', config.save_dir)
    centroids.to_csv(config.save_dir+'centroids.csv', index=False)
    listings.to_csv(config.save_dir+'listings.csv', index=False)
    
    try: assert sorted(centroids.centroid_group.unique()
           ) == sorted(listings.centroid_group.unique())
    except: raise BaseException(
        "ERROR: Inconsistent centroid groups between datasets.\rTry clearing cache & try again!")

    print(f"Completed in {round(time.perf_counter()-start)} seconds")
    print(' ', '—'*30)
    print('—'*60)



        



        