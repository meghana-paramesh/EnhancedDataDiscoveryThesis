import xml.etree.ElementTree as ET
from MetaData import MetaData

def get_maxar_meta_data(filename, image_file):
    maxar_tree = ET.parse(filename)
    maxar_root = maxar_tree.getroot()
    required_tag = ""

    for tags in maxar_root[0]:

        if tags.tag == "GENERATIONTIME":
            time = tags.text
        if tags.tag.startswith("BAND_"):
            required_tag = tags
            break

    for tags in required_tag:
        if tags.tag == "ULLON":
            ullon = tags.text

        if tags.tag == "ULLAT":
            ullat = tags.text

        if tags.tag == "URLON":
            urlon = tags.text

        if tags.tag == "URLAT":
            urlat = tags.text

        if tags.tag == "LRLON":
            lrlon = tags.text

        if tags.tag == "LRLAT":
            lrlat = tags.text

        if tags.tag == "LLLON":
            lllon = tags.text

        if tags.tag == "LLLAT":
            lllat = tags.text

    return MetaData(time, float(ullon), float(ullat), float(urlon), float(urlat), float(lrlon), float(lrlat), float(lllon), float(lllat), image_file)


def get_planet_meta_data(filename, image_file):
    # Parse the XML file
    tree = ET.parse(filename)
    root = tree.getroot()

    # Namespace for the XML elements
    namespace = {'ps': 'http://schemas.planet.com/ps/v1/planet_product_metadata_geocorrected_level'}

    # Find the element containing the coordinates
    ullat = root.find('.//ps:topLeft/ps:latitude', namespace).text
    ullon = root.find('.//ps:topLeft/ps:longitude', namespace).text

    urlat = root.find('.//ps:topRight/ps:latitude', namespace).text
    urlon = root.find('.//ps:topRight/ps:longitude', namespace).text

    lrlat = root.find('.//ps:bottomRight/ps:latitude', namespace).text
    lrlon = root.find('.//ps:bottomRight/ps:longitude', namespace).text

    lllat = root.find('.//ps:bottomLeft/ps:latitude', namespace).text
    lllon = root.find('.//ps:bottomLeft/ps:longitude', namespace).text

    # Namespace for the XML elements
    namespace = {'eop': 'http://earth.esa.int/eop'}

    # Find the element containing the acquisition date
    acquisition_date_element = root.find('.//eop:acquisitionDate', namespace)

    # Extract the acquisition date
    acquisition_date = acquisition_date_element.text

    return MetaData(acquisition_date, float(ullon), float(ullat), float(urlon), float(urlat), float(lrlon), float(lrlat), float(lllon), float(lllat), image_file)
