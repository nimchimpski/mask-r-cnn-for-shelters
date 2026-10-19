import xml.etree.ElementTree as ET

def extract_camp_kml(input_kml, output_kml, camp_pattern):
    """Extract a single camp from multi-camp KML by name filter."""
    tree = ET.parse(input_kml)
    root = tree.getroot()
    
    # Namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    ET.register_namespace('', 'http://www.opengis.net/kml/2.2')
    ET.register_namespace('gx', 'http://www.google.com/kml/ext/2.2')
    ET.register_namespace('atom', 'http://www.w3.org/2005/Atom')
    
    # Find folder
    folder = root.find('.//kml:Folder', ns)
    
    # Keep only Placemarks matching the pattern
    placemarks_to_keep = []
    for pm in folder.findall('kml:Placemark', ns):
        name = pm.find('kml:name', ns).text
        if camp_pattern in name:
            placemarks_to_keep.append(pm)
    
    # Remove all, then add back only matching
    for pm in folder.findall('kml:Placemark', ns):
        folder.remove(pm)
    for pm in placemarks_to_keep:
        folder.append(pm)
    
    # Write
    tree.write(output_kml, encoding='utf-8', xml_declaration=True)

# Extract
extract_camp_kml(
    'data/kutupalong/Camp_08E/CXB_New_camp_outline_KML_20180528.kml',
    'Camp_01E_outline.kml',
    '01E'
)

# Rename folder
with open('Camp_01E_outline.kml', 'r') as f:
    kml = f.read()
kml = kml.replace('<name>CXB_New_camp_outline_20180528</name>', '<name>Camp 01E</name>', 1)
with open('Camp_01E_outline.kml', 'w') as f:
    f.write(kml)