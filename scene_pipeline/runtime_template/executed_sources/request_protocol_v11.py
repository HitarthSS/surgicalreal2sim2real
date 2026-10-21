"""Lossless observation packing and explicit polygon coordinate construction.

This is a request-protocol revision, not a relaxation of reconstruction gates.
Full states and raw model responses remain saved by the orchestrator.
"""
from collections import Counter
from copy import deepcopy
import json
import math

MAX_PROMPT_CHARACTERS = 28000
ENCODING = ('Records {"$sN":[values]} use the ordered field names in schemas.sN. '
            'String values @vN refer to values.@vN, recursively. All other lists are literal lists. '
            'Expand references before interpreting evidence. No candidates, fields or numbers are omitted; '
            'IDs inside expanded records are the original executable IDs. Do not return encoding aliases as IDs.')


def compact_json(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def pack_observation(value):
    """Reversible shared-field records plus repeated-value references.

    The caller retains the established six-decimal observation rounding. Values
    in this function are exact, including missing fields versus explicit null.
    Alias allocation checks *all* original strings so literal aliases survive.
    """
    counts, originals, literal_strings = Counter(), {}, set()

    def count(item):
        token = compact_json(item)
        if len(token) >= 18:
            counts[token] += 1
            originals[token] = item
        if isinstance(item, str):
            literal_strings.add(item)
        elif isinstance(item, dict):
            for child in item.values():count(child)
        elif isinstance(item, list):
            for child in item:count(child)

    count(value)
    schemas, aliases, values = {}, {}, {}

    def encode(item, expand=False):
        token = compact_json(item)
        if not expand and counts[token] > 1 and len(token) >= 18:
            if token not in aliases:
                alias = '@v'+str(len(aliases))
                while alias in literal_strings or alias in values:alias += 'x'
                aliases[token] = alias
                # A subtree only references strictly smaller subtrees: no cycles.
                values[alias] = None
                values[alias] = encode(item, expand=True)
            return aliases[token]
        if isinstance(item, dict):
            keys = tuple(item)
            sid = schemas.setdefault(keys, 's'+str(len(schemas)))
            return {'$'+sid:[encode(child) for child in item.values()]}
        if isinstance(item, list):return [encode(child) for child in item]
        return item

    observation = encode(value)
    packed = dict(encoding=ENCODING, observation=observation,
                  schemas={sid:list(keys) for keys,sid in schemas.items()}, values=values)
    if unpack_observation(packed) != value:
        raise ValueError('Lossless observation round-trip failed')
    return packed


def unpack_observation(packed):
    def decode(item):
        if isinstance(item, str) and item in packed['values']:
            return decode(packed['values'][item])
        if isinstance(item, list):return [decode(child) for child in item]
        if isinstance(item, dict):
            if len(item) != 1:raise ValueError('Encoded record must have one schema tag')
            tag, row = next(iter(item.items()))
            fields = packed['schemas'][tag[1:]]
            if len(row) != len(fields):raise ValueError('Record width mismatch')
            return dict(zip(fields, [decode(child) for child in row]))
        return item
    return decode(packed['observation'])


def polygon_contract(image_sizes):
    return dict(coordinate_system='original full-frame pixel centers; top-left (0,0); x right, y down',
        views={side:dict(width=w,height=h,x_range=[0,w-1],y_range=[0,h-1])
               for side,(w,h) in image_sizes.items()},
        request={'polygons':{'left':[[[0,300],[60,310],[60,350],[0,350]]], 'right':[]}},
        format='0–2 polygons per view; each a list of 3–16 [x,y] finite numeric pairs; both view keys required. '
               'Example illustrates shape only, never actual object coordinates. Each polygon must enter a frame boundary. '
               'No normalized coordinates, boxes or vertex objects. Omit a polygon where no instrument is supported.',
        border_convention='Use width-1/height-1 for the last pixel. Legacy exact width/height edges are mapped '
                          'to the last pixel and logged; all other out-of-range coordinates are rejected.',
        unchanged_gates='per-polygon area 256 pixels to 30% of frame, at most two instruments per view; '
                        'observed needle/thread pixels remain protected; polygons are uncertain hypotheses')


def construct_polygon_request(parameters, image_sizes):
    """Validate syntax; only canonicalize exact outer extent to border pixel.

    No clipping arbitrary overflow, vertex deletion, contour interpolation or
    silent normalized-coordinate conversion. Raster/area/support gates run in
    the unchanged shared polygon tool after this construction step.
    """
    if (not isinstance(parameters, dict) or set(parameters) != {'polygons'}
        or not isinstance(parameters['polygons'], dict)
        or set(parameters['polygons']) != {'left','right'}):
        raise ValueError('Polygon request must be {polygons:{left:[polygons],right:[polygons]}}')
    result = deepcopy(parameters);changes = []
    for side in ('left','right'):
        w,h = image_sizes[side]
        if type(w) is not int or type(h) is not int or min(w,h) < 2:
            raise ValueError('Invalid original image dimensions')
        polygons = result['polygons'][side]
        if not isinstance(polygons,list) or len(polygons) > 2:
            raise ValueError(side+': at most two polygons required')
        for pi,polygon in enumerate(polygons):
            if not isinstance(polygon,list) or not 3 <= len(polygon) <= 16:
                raise ValueError(f'{side} polygon {pi}: require 3–16 vertices')
            for vi,point in enumerate(polygon):
                if not isinstance(point,list) or len(point) != 2:
                    raise ValueError(f'{side} polygon {pi} vertex {vi}: require [x,y]')
                for axis,extent in enumerate((w,h)):
                    coord = point[axis]
                    if type(coord) not in (int,float) or not math.isfinite(coord):
                        raise ValueError(f'{side} polygon {pi} vertex {vi}: finite numbers required')
                    if coord == extent:
                        point[axis] = extent-1
                        changes.append(dict(view=side,polygon=pi,vertex=vi,axis='xy'[axis],
                                            original=coord,canonical=extent-1))
                    elif not 0 <= coord <= extent-1:
                        raise ValueError(f'{side} polygon {pi} vertex {vi}: {"xy"[axis]}={coord} '
                                         f'outside pixel range 0..{extent-1}; no arbitrary clipping')
    return result, dict(coordinate_changes=changes, geometry_checks_relaxed=False,
                        convention='exact outer image extent to last pixel only',
                        raw_request=parameters, canonical_request=result)
