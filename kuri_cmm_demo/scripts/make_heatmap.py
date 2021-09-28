import os
from xml.etree import ElementTree as ElTree
from warnings import warn

from svgpathtools.parser import parse_transform, parse_path
import numpy as np
import matplotlib.pyplot as pl
import scipy.stats as st

svg_el = "{http://www.w3.org/2000/svg}svg"
image_el = "{http://www.w3.org/2000/svg}image"
text_el = "{http://www.w3.org/2000/svg}text"
circle_el = "{http://www.w3.org/2000/svg}circle"
line_el = "{http://www.w3.org/2000/svg}line"
poly_el = "{http://www.w3.org/2000/svg}polygon"
path_el = "{http://www.w3.org/2000/svg}path"
group_el = "{http://www.w3.org/2000/svg}g"
tspan_el = "{http://www.w3.org/2000/svg}tspan"


def get_text_from_group(group):
    # Inkscape tucks things in a tspan. Check that first
    text = group.find(".//{}".format(tspan_el))
    if text is None:
        text = group.find(".//{}".format(text_el))
    if text is None:
        return None
    return text.text


def get_point(element):
    return float_s3(element.attrib["cx"]), float_s3(element.attrib["cy"])


def float_s3(string):
    return round(float(string), 3)


def np_point_to_tuple(np_point):
    return tuple(np_point[:2, 0])


def apply_transform(point, transform):
    if isinstance(point, tuple):
        # Convert to homogeneous form
        point = np.array([[point[0], point[1], 1]])
    return np_point_to_tuple(np.matmul(transform, point.transpose()))


def get_transform(element):
    if "transform" in element.attrib:
        return parse_transform(element.attrib["transform"])
    else:
        return np.identity(3)


def is_line(path_part):
    from svgpathtools import Line
    return isinstance(path_part, Line)


def extract_line_from_path(path, transform=None):
    """
    Treat a path as a line-segment and extract end points. Throws
    if the path isn't a line.

    :param path:
    :param transform: the transform to apply to the coordinates
    :return: tuple of line-segment start and end coordinates
    """
    path_geom = parse_path(path.attrib["d"])

    if len(path_geom) == 1 and is_line(path_geom[0]):
        line = path_geom[0]
        # We assume line starts at origin and points towards the second point
        start_coord = (float_s3(line.start.real), float_s3(line.start.imag))
        end_coord = (float_s3(line.end.real), float_s3(line.end.imag))
        return apply_transform(start_coord, transform), apply_transform(end_coord, transform)
    else:
        raise RuntimeError()


def create_heatmap():
    annotation_path = "../share/floorplan.svg"

    if not os.path.isfile(annotation_path):
        # No annotations to load. Since you're trying to load annotations, this is probably an error of some sort
        warn("No annotation file found at {}".format(annotation_path))
        return None

    with open(annotation_path) as test_svg:
        svg_data = test_svg.readlines()
        svg_data = " ".join(svg_data)

    points = process_svg(svg_data)
    x, y = [], []
    for point in points:
        x.append(point[1][0])
        y.append(point[1][1])
    xmin, xmax = 0, 995
    ymin, ymax = 0, 311
    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:0.25, ymin:ymax:0.25]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values, bw_method=0.15)
    f = np.reshape(kernel(positions).T, xx.shape)
    norm = pl.Normalize(vmin=f.min(), vmax=f.max())
    # Contourf plot
    # cfset = ax.contourf(xx, yy, f, cmap='Greens')
    im = np.flipud(np.rot90(norm(f)))
    im = pl.get_cmap("Greens")(im)
    pl.imsave('heatmap_kernel.png', im)

    return


def create_heatmap_scatter():
    annotation_path = "../share/floorplan.svg"

    if not os.path.isfile(annotation_path):
        # No annotations to load. Since you're trying to load annotations, this is probably an error of some sort
        warn("No annotation file found at {}".format(annotation_path))
        return None

    with open(annotation_path) as test_svg:
        svg_data = test_svg.readlines()
        svg_data = " ".join(svg_data)

    points = process_svg(svg_data)
    x, y = [], []
    for point in points:
        x.append(point[1][0])
        y.append(point[1][1])
    xmin, xmax = 0, 995
    ymin, ymax = 0, 311

    fig, ax = pl.subplots(figsize=(51.36, 16.47), dpi=100)

    ax.scatter(x, y, c='g', alpha=0.25, s=3600, marker='o', linewidth=0)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    fig.gca().set_aspect('equal', adjustable='box')
    fig.gca().invert_yaxis()
    pl.axis('off')
    fig.savefig("heatmap_scatter.png", bbox_inches='tight', pad_inches=0)


def process_svg(svg_data):
    """
    Extracts annotations from SVG data. See documentation for an explanation of
    how annotations are expected to be structured.

    :param svg_data: string containing SVG data
    :return: extracted point, pose, region and door annotations
    """
    tree = ElTree.fromstring(svg_data)
    parent_map = {c: p for p in tree.iter() for c in p}
    point_annotations = tree.findall(".//{}".format(circle_el))
    point_names = tree.findall(".//{}/../{}".format(circle_el, text_el))
    circle_groups = tree.findall(".//{}[{}]".format(group_el, circle_el))
    # The point annotations we care about have just a dot and a text label
    point_groups = filter(lambda g: len(list(g)) == 2, circle_groups)
    point_parents = map(parent_map.__getitem__, point_annotations)
    points = process_point_annotations(point_names, point_annotations, point_parents)

    extra_points = []

    for group in point_groups:
        name = get_text_from_group(group)
        transform = get_transform(group)

        circle = group.find(".//{}".format(circle_el))

        circle_center = get_point(circle)
        pixel_coord = apply_transform(circle_center, transform)
        extra_points.append((name, pixel_coord))

    # points += extra_points

    return points


def process_point_annotations(point_names, point_annotations, point_groups):
    points = []
    for point, text, parent in zip(point_annotations, point_names, point_groups):
        name = text.text
        pixel_coord = apply_transform(get_point(point), get_transform(parent))
        points.append((name, pixel_coord))
    return points


if __name__ == "__main__":
    create_heatmap()
    create_heatmap_scatter()
