import os
import math
import json
import config
import folium
import numpy as np
import pandas as pd

# to normalize & standardize opacity
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm

# applies host-group decoder ALWAYS ON
def decoder_mode(active):
    print('—'*60)
    if active:
        config.data_dir = 'data_cleaned/corp_grouped/'
        config.host_decoder = True
        print('> Corp group decoder mode ACTIVE')
    else:
        config.data_dir = 'data_cleaned/host/'
        config.host_decoder = False
        raise("DEPRECATD")
        print('> Corp group decoder mode is OFF')
    print('  > Data directory set to', config.data_dir)


# ———————————————————————————————————————— 
# —————————— LOAD DATA FUNCTIONS —————————

def load_adco_data():
    print('-'*60)
    acdo_file = 'data_raw/STRbldgsGPSplus.csv'
    print(f'> Loading ADCO STR building database... {acdo_file}')
    globals()['acdo_data'] = pd.read_csv(acdo_file)
    global adco_data
    acdo_data['latitude'] = acdo_data['latlong'].apply(lambda x: x.split(',')[0]).astype(float)
    acdo_data['longitude'] = acdo_data['latlong'].apply(lambda x: x.split(',')[1]).astype(float)
    print(f'  > {len(acdo_data)} STR Buildings found')

    # FOR ADCO MAPPING:
def isolate_probable_bldgs(ctrd_row, bldgs):
    
    lat_diff = abs(bldgs['latitude'] - ctrd_row.mean_latitude)
    lng_diff = abs(bldgs['longitude'] - ctrd_row.mean_longitude)
    
    bldgs['geo_diff'] = (lat_diff + lng_diff) / 2
    
    search_rad = .0005
    max_search_rad = .003
    nearby = bldgs[bldgs.geo_diff<search_rad].copy()
    while len(nearby)<2 and search_rad<max_search_rad:
        search_rad += .0001
        nearby = bldgs[bldgs.geo_diff<search_rad].copy()
    
    return nearby
    
    
def load_data():
    print('—'*60)
    centroids_file = config.data_dir+'centroids.csv'
    listings_file = config.data_dir+'listings.csv'

    centroids = pd.read_csv(centroids_file)
    for c in centroids.columns:
         if '_dict' in c:
            centroids[c] = centroids[c].apply(lambda x: json.loads( x.replace("'", '"') ) )

            hosts = centroids.host_id.unique()
    c_groups = centroids.centroid_group.unique()
    print(f'> Centroid data loaded from {centroids_file}')
    print('  >', len(hosts), 'hosts |', len(c_groups), 'centroids |', centroids.count_listings.sum(), 'listings')

    listings = pd.read_csv(listings_file)
    listings.rename(columns={'host_name':'host'}, inplace=True)
    for row in listings.index:
        if listings.loc[row, 'centroid_group'] not in c_groups:
            listings.drop(row, inplace=True)
    listings.reset_index(drop=True, inplace=True)
    print(f'> Listings data loaded from {listings_file}')
    print('  >', listings.host_id.nunique(), 'hosts |',
        listings.centroid_group.nunique(), 'centroids |',
        len(listings), 'listings')
    print('—'*30)
    
    print('> Calculating centroids map element radius') 
    centroids['radius'] = (centroids['avg_listing_distance'
                         ]+centroids['std_listing_distance']
                         )*0.3048 # convert feet -> meters for folium
    centroids.radius = centroids.radius.apply(lambda x: max(50, x)) # set minimum centroid circle size  
    
    print('> Calculating centroids map element opacity...') 
    scaler = MinMaxScaler()
    centroids['opacity'] = (scaler.fit_transform(centroids[['count_listings']].apply(np.log))*.2)+.4
    return centroids, listings # (centroids, listings)



# ———————————————————————————————————————— 
# —————————— DATA SPLIT FUNCTIONS ————————


def drop_sub3_centroids(centroids, df=False):
    tinies = centroids[centroids.count_listings<3].centroid_group.values
    if type(df)==bool and not df: df = centroids
    dropped = 0
    for row in df.index:
        if df.loc[row, 'centroid_group'] in tinies:
            df.drop(row, inplace=True)
            dropped +=1
    print(f'  > Dropped {dropped}', 'centroids' if len(df)==len(centroids) else 'listings')
    return df 


def big_host_focus(centroids, listings, min_size):
    print('—'*60)
    
    # host must have at least 1 centroid with min_size listings to remain
    keep_hosts = centroids[centroids.count_listings>=min_size].host_id.unique()
    print(f'> Identified {len(keep_hosts)}/{centroids.host_id.nunique()} big hosts (at least 1 centroid with >={min_size} listings)')
    l_olen = len(listings)
    c_olen = len(centroids)
        
    def drop_small_hosts(df):
        df.reset_index(drop=True, inplace=True)
        for row in df.index:
            if df.loc[row, 'host_id'] not in keep_hosts:
                df = df.drop(row)
        return df.reset_index(drop=True)
    centroids = drop_small_hosts(centroids)
    listings = drop_small_hosts(listings)

    print(f'  > Dropped {c_olen - len(centroids)} centroids ({l_olen - len(listings)} listings)')
    print(f'  > Remaining: {len(centroids)} centroids ({len(listings)} listings)')
    return centroids, listings 
    
    
    
# ———————————————————————————————————————— 
# ——UMBRELLA LOAD & CLEAN LOAD FUNCTION——— 

def prep_data(home=True):
    decoder_mode(True)
    load_adco_data()
    centroids, listings = load_data()
    if home:
        centroids, listings = big_host_focus(centroids, listings, 10)
        print('> Dropping tiny centroids...') 
        listings = drop_sub3_centroids(centroids, listings)
        centroids = drop_sub3_centroids(centroids)
        
    print('> Applying color keys to license statuses...')
    listings = color_license_status(listings)
    centroids = color_license_status(centroids)
    print('  > Done.')
    print('—'*60)
    return centroids, listings
    

    
    



# ———————————————————————————————————————— 
# ——— UNIVERSAL VARS & HELPER FUNCTIONS ——


def global_search(centroids, listings):
    # COMPILE SEARCH TERMS (UNIVERSAL) (with **all** interesting centroids)

    search_html = '' # to compile into

    # COMPILE HOSTS
    
    for host in listings.host_id.value_counts().index:
        h_data = listings[listings.host_id==host]
        host_name = h_data.host.values[0]
        host_ids = list(h_data.host_id.unique())
        aliases = ', '.join([h for h in h_data['host_alias'].dropna().astype(str).unique()])
        cluster_n = h_data.centroid_group.nunique()
        search_html += f"""
        <li onclick="location.href='/host/{host}.html'"
            class="search-items">
            <data = "{host_ids}">
            <p style='font-size:1em;color:white'>Host: <b style='color:#C5B358'>{host_name}</b></p>
            {f"<p style='font-size:.9em;color:white'>Aliases: <b>{aliases}</b></p>" if aliases else ''}
            <p style='font-size:.9em; color:white'>Clusters: <b>{cluster_n}</b> | Listings: <b>{len(h_data)}</b></p>
        </li>
        """
    # ADD CLUSTERS
    for row in centroids.index:
        c_data = centroids.loc[row]
        search_html += f"""
        <li onclick="location.href='/centroid/{c_data.centroid_group}.html'"
            class="search-items">
            <p style='font-size:1em;color:white'>Cluster: <b style='color:#448AFF'>{c_data.centroid_group}</b></p>
            <p style='font-size:.9em; color: white'>Host: <b>{c_data.host}</b> | Listings: <b>{c_data.count_listings}</b></p>
        </li>
        """
        
        
        
    config.search_html = f"""

                <p id="search-prompt" style="color: rgb(224, 224, 224); display: block;
                       font-size: 0.95em; margin: 0px 0px 5px; max-height: 150px;">
                    Search hosts & listings cluster: 
                </p>
                
                <div id="search-container">
                    <input id="searchbar" onkeyup="search_item()" tabindex="1"
                    type="text" name="search" placeholder="Search...">
                    
                    <div id='search-message-container'> 
                        <p id="search-instruct" style="display: block; font-size:.82em !important;
                            color:#448AFF;">
                                Click for more details:</p>
                        <p id="search-alert" style="display: block; font-size:.82em !important;
                                color:#F44336;">
                                    <b>No results</b></p>
                        <p id="search-init" style="display: block; font-size:.82em !important;
                                color:#E0E0E0;">
                                    <em>Search by account name, alias, host ID, or cluster group</em></p>
                    </div>
                    <ul id="search-list">
                        {search_html}
                    </ul>
               </div>



    """
        
    print('Created search terms')


# global license color keys <<<————————————
simple_color_dict = {
    'Active': '#2E7D32',
    'Expired/Void/Revoked': '#B71C1C',
    'Not verified': '#E65100',
    'None claimed':  '#455A64',
    'Not found (fabricated)': '#D32F2F', }

def color_license_status(df):
    
    global simple_color_dict 
    exempt_dict = {
        'Exemption: hospital contracts': '#F57F17',
        'Exemption: hotel/motel': '#FF8F00',
        'Exemption: executive suite': '#E65100' }

    # for listing status add color
    if 'status_dict' not in df.columns: # LISTINGS DATA
        for row in df.index:
            license = df.loc[row, 'license']
            status = df.loc[row, 'status']
            if 'Not verified' in str(status):
                df.loc[row, 'color'] = exempt_dict[license]
            else:
                df.loc[row, 'color'] = simple_color_dict[status]
                
    # for centroids
    else:
        for row in df.index:          # CENTROIDS DATA 
            licenses = list(df.loc[row, 'license_dict'].keys())
            statuses = list(df.loc[row, 'status_dict'].keys())
            if 'Not verified' in str(statuses[0]):
                for license in licenses:
                    try:
                        df.loc[row, 'color'] = exempt_dict[license]
                    except: pass
            else:
                df.loc[row, 'color'] = simple_color_dict[statuses[0]]
    return df



    
# ———————————————————————————————————————— 
# ————————— TOOLTIPS FUNCTIONS ——————————— 


def centroid_toolip(c_data, listings):
    multi_license = len(c_data.license_dict.items())>1
    if multi_license: license_list = 'Licenses (multiple):'
    else: license_list = 'License:'
    for license, count in c_data.license_dict.items():
        color = listings[listings.license==license].color.values[0]
        status = listings[listings.license==license].status.values[0]
        license_list += f"""<br><span style='padding-left:5px'>
                                {f'{count}x ' if multi_license else ''}<i style='font-size:.9em;color:{color}'
                                   class='fa fa-file-text'></i>
                                <b> {license}</b> | <span style='color:{color}'> {status} </span>"""
        license_list+= "</span>"
        
    t = folium.Tooltip(f"""
                        <p style='font-size:1.25em; margin:0; color:#448AFF'>Cluster <b>{c_data.centroid_group}</b></p>
                        Host: <b>{c_data.host} </b> | Listings: <b>{c_data.count_listings}</b>
                        <br>{license_list}
                        <br><span style='font-size:.92em; text-align:center; color:#448AFF'>
                            <b>Click for more details &#8594;</b></span>
                        """)
    return t 


def listing_toolip(l_data):

    t = folium.Tooltip(f"""
                        <p style='font-size:1.25em; margin:0; color:#BF0000'>Listing <b>{l_data.id}</b></p>
                        <span style='color:#448AFF'>Cluster: <b>{l_data.centroid_group}</b></span>
                        Host: <b>{l_data.host} </b>
                        
                        <br>License:<br>
                             <span style='padding-left:5px'>
                                 <i style='font-size:.9em; color:{l_data.color}';
                                       class='fa fa-file-text'></i>
                        <b> {l_data.license}</b> | <span style='color:{l_data.color}'> {l_data.status} </span>
                            </span>
                        <br><span style='font-size:.92em; text-align:center; color:#448AFF'>
                        <b>Click for centroid details &#8594;</b></span>
                        """)
    return t



def adco_tooltip(row): # NOT IN USE
    
    if str(row['Bldg name'])!='nan':
        html = f"<b>{row['Bldg name']}<br>"
    else: html = ''       
    html += str(row.GPSaddress)+'</b><br>'
    
    details = []
    if row['Bldg units']: details.append(f"Units: {row['Bldg units']}")
    if row['Bldg flrs']: details.append(f"Floors: {row['Bldg flrs']}")
    if row['Source']: details.append(f"Source: {row['Source']}")
    if len(details)>1: html+= ' | '.join(details)+'<br>'
        
    bldg_hosts = []
    for col, val in zip(row.index, row.values):
        if 'Bldg' not in col and str(val)==str(1.0):
            bldg_hosts.append(col)
            
    if len(bldg_hosts)<1: bldg_hosts.append('unknown')
    html += "<b>Hosts: </b>"+', '.join(bldg_hosts)
    
    return folium.Tooltip(html)




# ———————————————————————————————————————— 
# ———————————— MAP ANNOTATIONS ———————————

def add_updated(m, listings):
    updated = f"""
         <div style="
         position: fixed; z-index:699; 
         bottom: 16px; right: 0; width: auto; height: auto; 
         background-color:white;
         border-bottom:1px solid #DEDEDE;
         opacity: .7;
         padding: 3px 8px;
         font-size:12px">
            <span style='margin: 0px;'>
                <a target='blank' href='insideairbnb.com'>InsideAirbnb</a>:
                    {listings.last_scraped.max()} | STR registry: 2020-08-05
            </span>
          </div> """
    m.get_root().html.add_child(folium.Element( updated ))
    return m





# ———————————————————————————————————————— 
# ——————————— WEB PAGE SIDEBAR ———————————

# all of this for the license status legend
def get_status_vcs(data):
 
    vcs = data.status.replace('Expired', 'Expired/Void/Revoked'
                    ).replace('Void', 'Expired/Void/Revoked'
                    ).replace('Revoked', 'Expired/Void/Revoked'
                    ).replace('Inactive', 'Expired/Void/Revoked'
                    ).replace('fabricated', 'Not found (fabricated)'
                    ).replace('none claimed', 'None claimed'
                    ).replace('not verified', 'Exemption claimed'
              ).value_counts().sort_index()
    global simple_color_dict
    out = vcs.to_frame().reset_index()
    out = out.rename(columns={'index':'status', 'status':'count'})
    out['color'] = out['status'].map(simple_color_dict)
    return out

def make_status_legend(listings):
    status_vcs = get_status_vcs(listings).replace('Not verified', 'Exemption claimed')
    status_key_html = """<div class='listings-status-legend'>"""
    status_vcs = status_vcs.sort_values('count', ascending=False)
    for row in status_vcs.index:
        status_info = status_vcs.loc[row]
        row_html = f"""<p style='font-size:14px; margin:5px 0;'>                            
                            
                            <span id='status-icon'
                                style="background-color:{status_info['color']}">
                            </span>
                            <span>
                                x {status_info['count']} &mdash;
                                <b>{status_info.status}</b>
                            </span>
                        </p>
                        """
        status_key_html += row_html
    return status_key_html + '</div>'





# ———————————————————————————————————————————————————————————————————————————————————————————————————— 
# ——————————————————————————————————— MAP HTML  FUNCTIONS  ———————————————————————————————————————————

      

def map_all_centroids(centroids, listings):

    # base map
    m = folium.Map(location=[42.35, -71.07], tiles=None, zoom_start=14)
    folium.TileLayer('cartodbpositron', control=False).add_to(m) # hide "all" button

    # CENTROID AREA CIRCLES
    for row in centroids.sort_values('radius', ascending=False).index:
        c_data = centroids.loc[row]
        folium.vector_layers.Circle(
                            [c_data.mean_latitude, 
                             c_data.mean_longitude],
                            color = '#448AFF',
                            fill = True,
                            weight = 2,
                            fill_color = c_data.color,
                            radius =  c_data.radius,
                            opacity = c_data.opacity+.3,
                            fill_opacity = c_data.opacity,
                            tooltip = centroid_toolip(c_data, listings)
                            #popup = centroid_popup(c_data),
            ).add_to(m)

    # LISTING MARKERS
    listing_fg = folium.FeatureGroup( 
            name="""<span style='font-size:15px'>
                        show individual listings
                        <i class='fa fa-dot-circle-o'
                           style='color:red; margin: 0 3px;'></i>
                    </span>""",
            show=False,
            control=True) # for user layer control
    m.add_child(listing_fg)
    for row in listings.index:
        l_data = listings.loc[row]
        folium.vector_layers.Circle(
                            [l_data.latitude, 
                             l_data.longitude],
                            color = 'red',
                            fill_opacity = .8,
                            weight=2.5,
                            opacity = .9,
                            fill_color = l_data.color,
                            radius = 10,
                            tooltip = listing_toolip(l_data)
            ).add_to(listing_fg)


    # CENTROID CENTER MARKERS
    for row in centroids.sort_values('radius', ascending=False).index:
        c_data = centroids.loc[row]
        folium.vector_layers.Circle(
                            [c_data.mean_latitude, 
                             c_data.mean_longitude],
                            color = '#448AFF',
                            fill = True,
                            weight = 4,
                            fill_color = '#448AFF',
                            radius =  6,
                            opacity = .7,
                            fill_opacity = 1,
                            tooltip = centroid_toolip(c_data, listings)
                            #popup = centroid_popup(c_data),
            ).add_to(m)


    folium.LayerControl(collapsed=False).add_to(m)
    
    m = add_updated(m, listings)
    return m


# ———————————————————————————————————————————————————————————————————————————————————————————————————— 
# ———————————————————————————————— PAGE GENERATOR FUNCTIONS  —————————————————————————————————————————




def all_centroids_page(m, centroids, listings):
    
    
    print('—'*60)
    print('> Generating home page map...')
    mf_dir = 'this_map_file.html'
    m_file = m.save(mf_dir, close_file=False)


    # HOME PAGE CSS
    sidebar_width = '400px' # var
    style = """
    body {
        padding:0;
        margin:0;
        height:100vh;
        overflow: auto;
    }

    h1, h2, h3, h4, p, pg, a {
        font-family: "Helvetica", Arial, sans-serif;

    }

    a {
        color: #E3F2FD;
        text-decoration:none
    }
    a:hover {
        text-decoration:underline
    }

    h1 {
        font-size:30px;
        color:#64B5F6;
        margin: 8px 0 4px;

    }

    h2 {
        font-size:24px;
        line-height:1.4em;
        color:white;
        font-weight:400px;
        margin: 8px 0 8px;
    }

    h3 {
        font-size:18px;
        color:white;
        margin: 0 0 12px;
        font-weight:400px;
        font-weight:500px;
    }

    h4 {
        font-size:20px;
        color:white;
        margin: 0 0 10px;
        font-weight:500px;
    }

    p {
        font-size:16px;
        line-height:1.4em;
        color:white;
        margin: 0 0 8px;
    }

    pg {
        color:#E0E0E0;
    }

    hr {
        border: .6px solid white;
        margin: 15px 0
    }


    #page-container {
        padding: 0;
        display:block;
        height: 100%;
        width: 100vw;
        box-sizing: border-box;
    }

    .sidebar-container {
        position:fixed;
        display: block;
        height: 100vh;
        width:"""+sidebar_width+""";
        overflow:auto;
    }

    .nav-banner {
        position:fixed;
        top:0;
        left:0;
        z-index:9;
        margin:0;
        width:"""+sidebar_width+""";
        background-color:#191919;
    }
    
    .nav-banner p {
        margin:4px 8px;
    }

    .sidebar {
        width:"""+sidebar_width+""";
        top: 25px;
        position:fixed;
        height:calc(100vh - 25px);
        display:block;
        background-color: #2b2b2b;
        #border: 5px solid #4a4a4a;
        box-sizing: border-box;
        overflow-y: auto;
        padding:30px;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.4);
    }

    .listings-status-legend {
        margin-top:12px;
        padding-left: 12px;
        border-left:  2px solid white
    }

    #status-icon {
        height:14px;
        width:14px;
        display:inline-block;
        border: 2px solid white;
        border-radius:50%;
        margin: 5px 5px 0 0;
    }

    #map-container {
        position:fixed;
        top:0;
        right:0;
        display: block;
        height:100vh;
        width: calc(100% - """+sidebar_width+""")
    }

    iframe {
        position:fixed;
        border:none;
        right:0;
        top:0;
        height:100%;
        width: calc(100% - """+sidebar_width+""")
    }

#search-container {
    border: 2px solid #448AFF;
    height: auto;
    margin:0;
    width:100%;
    padding:0;
    border-radius:3px;
    background-color:#424242;
    transition: .2s ease-in-out;    
    }
    
#search-list{
    width: calc(100% - 24px) !important
}

#searchbar {
        background-color:#424242;
        width: calc(100% - 20px) !important; 
        border-radius: 5px;
        border: 0px !important;
        padding:10px;
        margin:0;
        font-size:16px;
        color:white;
       } 
       
      

::placeholder {
        font-size:18px;
        font-weight: bold;
        color:#82B1FF;
        opacity: 1 !important; 
        margin:0
    }
    
#search-message-container{
max-width:265;
}

#search-list{ 
    display:block;
    width: calc("""+sidebar_width+""" - 30px);
    margin:0;
    font-size: 16px;
    max-height:250px;
    list-style-type: none;
    padding: 0 15px;
    overflow:auto;
    transition: .2s ease-in-out;
   }

.search-items { 
   color: black;
   display: none;
   padding:10px 0;
   cursor: pointer;
   border-top: 1px solid white;
  } 
  
.search-items:hover {
    boxShadow = inset 0 0 34px rgba(0,0,0,0.4);
}

.search-items p {
    margin-bottom:2px
}

#search-prompt {
    overflow: hidden;
    max-height: 150px;
    transition: .2s;
  }

#search-instruct {
    overflow: hidden;
    max-height: 0;
    transition: .2s;
    padding: 0 8px;
    margin:0;

  }
#search-alert {
    overflow: hidden;
    max-height: 0;
    transition: .2s;
    padding: 0 8px;
    margin:0;

  }
#search-init {
    overflow: hidden;
    max-height: 0;
    transition: .2s;
    padding: 0 8px;
    margin:0;
  }

      
#how-collapse-btn {
  background-color: transparent;
  color: white;
  cursor: pointer;
  padding: 8px 0;
  width: 100%;
  border: none;
  text-align: left;
  outline: none;
  font-size: 15px;  
}

#how-collapsed {
  padding: 5px 15px;
  border-left: 1px double white;
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.2s ease-out;
}


    """


    # HOME PAGE JAVASCRIPT
    script = """
        document.getElementById("how-collapse-btn"
            ).addEventListener("click", function() {
            var content = this.nextElementSibling;
            if (content.style.maxHeight){
              content.style.maxHeight = null;
              this.innerHTML = "[ + ] Learn more... ";
            } else {
              content.style.maxHeight = content.scrollHeight + "px";
              this.innerHTML = "[ - ] less ";
            } 
        });
        </script>
    <script>
    

    
    document.getElementById('searchbar').addEventListener("click",
        function(){
                let x = document.getElementsByClassName('search-items');
                for (i = 0; i < 10; i++) {  
                    x[i].style.display="list-item"; 
                } 
                document.getElementById('search-list').style.maxHeight = "271px";
                document.getElementById('search-list').style.padding = "0 15px 15px";
                let msg_init = document.getElementById('search-init'); 
                msg_init.style.maxHeight = "150px";
                msg_init.style.margin = "8px";
                document.getElementsByClassName('search-list').style.boxShadow = "inset 0 0 20px rgba(0,0,0,0.6)";
        }
    )
    
    
    document.getElementById('searchbar').addEventListener("focusout",
        function(){
            let input = document.getElementById('searchbar').value;
            if (input==""){
                document.getElementById('search-list').style.maxHeight = "0";
                document.getElementById('search-list').style.padding = "0 15px";
            
                let x = document.getElementsByClassName('search-items');
                for (i = 0; i < x.length; i++) {  
                    x[i].style.display="none"; 
                } 
            }
            let msg_init = document.getElementById('search-init'); 
            msg_init.style.maxHeight = "0";
            msg_init.style.margin = "0";
            document.getElementsByClassName('search-list').style.boxShadow = "inset 0 0 0 rgba(0,0,0,0.5)";
            
        }
    )

    
    function search_item() { 
        
        let sb = document.getElementById('searchbar')
        let input = document.getElementById('searchbar').value;
        input=input.toLowerCase(); 
        let searchList = document.getElementById('search-list');

        let x = document.getElementsByClassName('search-items'); 
        
        let instruct = document.getElementById('search-instruct'); 
        let alert = document.getElementById('search-alert'); 
        let s_init = document.getElementById('search-init'); 
        
        let msg_init = document.getElementById('search-init'); 
        msg_init.style.maxHeight = "0";



        var visible = 0
        
        if (sb !== null && sb.value === "") {
        
            for (i = 0; i < 10; i++) {  
                x[i].style.display="list-item"; 
                    visible += 1
            } 
        }
        else {
            for (i = 0; i < x.length; i++) {  
                if (x[i].innerHTML.toLowerCase().includes(input)) { 
                    x[i].style.display="list-item"; 
                    visible += 1
                } 
                else { 
                    x[i].style.display="none";
                } 
             } 
        }

        if (sb.value !== "") {
                if (visible==0) {
                    instruct.style.maxHeight = "0";
                    instruct.transition = ".2";
                    instruct.style.margin = "0";
                    s_init.style.maxHeight = "0";
                    s_init.transition = ".2s";
                    s_init.style.margin = "0";
                    
                    alert.style.maxHeight = "150px";
                    alert.style.margin = "8px 0 5px";
                   
                    
                    searchList.style.padding = "0";
                    searchList.style.maxHeight = "0";
                    

                    }
                else {
                    instruct.innerHTML = "<b>" + visible.toString() + "</b> results - click  for more details:" ;
                    instruct.style.maxHeight = "150px";
                    instruct.transition = "0";
                    instruct.style.margin = "8px 0 5px";
                    
                    
                    s_init.style.maxHeight = "0";
                    s_init.transition = ".2s";
                    s_init.style.margin = "0";
                    alert.style.maxHeight = "0";
                    alert.transition = ".2s";
                    alert.style.margin = "0";
                    
                    searchList.style.padding = "0 15px 15px ";
                    searchList.style.maxHeight = "271px";                    
                    }
 
            
          }         
        else {
            instruct.style.maxHeight = "0";
            instruct.transition = ".2s";
            instruct.style.margin = "0"
            alert.style.maxHeight = "0";
            alert.transition = ".2s";
            alert.style.margin = "0";
            s_init.style.maxHeight = "0";
            s_init.style.margin = "0";
            searchList.style.maxHeight = "0";
            searchList.style.padding = "0";
                    }       
        }


    """


    # HOME PAGE SIDEBAR LAYOUT
    
    site_title = """    
    <p style='color:white;font-size:1.7em;'>unofficial <b>STR Monitor</b></p>
    <p style='margin:0 0 10px;'>Boston Metro Area</p>"""

    sidebar = f"""

        {site_title}
        <hr style='margin:10px 0 8px'>



        {config.search_html}

        <hr style='border-color:#4a4a4a; style='margin:20px'>


        <pg style='font-size:16px; margin-bottom:0'>
        <h2 style='color:#448AFF; font-size:22px'>High-confidence clusters </h2>

        <p style='color:white'>
        
        Airbnb listings are clustered algorithmically into groups based on host, license and semi-anonymous geographic data.
        In theory, each cluster corresponds to a building with multiple STR units (within a given range accuracy).
        </p>
        
        <a id="how-collapse-btn">[+] Learn more... </a>
        <div style='font-size.9em;color:white;' id="how-collapsed">
            <p style='margin-top:5px'>
            Dbscan clustering algorithms are applied over multiple phases to acquire final
            best-guess locations for suspected high-occupancy STR buildings. First listings 
            are split into unique host-license groups, because hosts often use the same license
            number for all the units in a building. In phase I, the clustering algorithm
            overestimates cluster granularity, generating many small groups.
            </p><p>
            In the second phase of clustering, we combine overlapping sub-groups owned by
            the same host or corporation group, accounting for cases where one host uses multiple
            license numbers within the same building. The end goal is identify distinct
            clusters of listings belonging to a given building.
            </p><p>
            In buildings with many assumed listings, we exploit the semi-anonymized public
            location data to approximate a guess for the actual building location by 
            taking the mean weighted coordinates of all the listings in a cluster. This strategy
            often leads directly to or near a building with confirmed STR units inside.
            </p><p>
            A repository containing the full stack for this analysis is available on GitHub:
            </p>
        </div>



        <hr style='border-color:#4a4a4a; style='margin:25px 0'>

        <h4 style='font-size:20px; margin-bottom:8px'>License Status Legend</h4>
        <pg style='margin-bottom:10px;'>Licenses claimed on {len(listings)} listings ({len(centroids)} clusters), cross-referenced to <a target='blank' href='https://www.boston.gov/departments/inspectional-services'>ISD</a> data
        </pg>

        {make_status_legend(listings)}

        <div style='font-size:.88em; margin-top:15px'>
            <pg>
            <b>Note:</b> The map on this page includes only the top {len(centroids)} clusters
            (with 10 or more listings) to highlight suspected STR-heavy buildings. For an overview
            of all known listings, <a href='/'>return to the home page</a></pg>
        </div>


    """


    # HOME PAGE PAGE HTML

    page = f"""
    <!DOCTYPE html>
    <html>
    <head>

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>{style}</style>

    <head>
    <body>

        <div id='page-container'>

            <div class='sidebar-container'>
                    <div class='nav-banner'>
                        <p>
                        <span style='cursor:pointer;'>
                            <a style='color:#448AFF' href='/'>Home</a> &#62; Clusters
                        </span>
                    </div>
                <div class='sidebar'>
                    {sidebar}            
                </div>
            </div>


            <div id='map-container'>
                <iframe id="map-sec"
                    title="Map Section"
                    height=100%
                    width=100%
                    src="{mf_dir}">
                </iframe>
            </div>

        </div>

        <script>
        {script}
        </script>

    </body>

    </html>
    """

    Html_file= open("site_data/clusters.html","w")
    Html_file.write(page)
    Html_file.close()
    
    
    
def run():
    print("GENERATING MAP OF TOP CENTROIDS")
    # FOR ALL PAGES
    centroids, listings = prep_data(False)
    global_search(centroids, listings) # UNIVERSAL SEARCH BAR
    # SPLICE FOR SMALLER DATA FOR HOME PAGE
    centroids, listings = prep_data()
    # CREATING MAIN (ALL TOP CENTROIDS MAP)
    m = map_all_centroids(centroids, listings)
    all_centroids_page(m, centroids, listings)



