from typing import Dict, List

def annotation_fix(annotations: Dict[str, List[Dict[str, int]]]) -> Dict:

    print("Fixing annotations...")

    new_annotations = {}

    for path, annotations in annotations.items():
        new_annotations[path] = []
        annos_to_process = annotations

        if len(annos_to_process) == 0:
            continue

        avg_x = sum([a["x_max"] - a["x_min"] for a in annos_to_process]) / len(annos_to_process)
        avg_y = sum([a["y_max"] - a["y_min"] for a in annos_to_process]) / len(annos_to_process)

        while len(annos_to_process) != 0:
            processing = []

            for i in range(len(annos_to_process)):
                if annos_to_process[i]["x_min"] > annos_to_process[i]["x_max"]:
                    annos_to_process[i]["x_min"],
                    annos_to_process[i]["x_max"] = \
                        annos_to_process[i]["x_max"], \
                        annos_to_process[i]["x_min"]

                if annos_to_process[i]["y_min"] > annos_to_process[i]["y_max"]:
                    annos_to_process[i]["y_min"],
                    annos_to_process[i]["y_max"] = \
                        annos_to_process[i]["y_max"], \
                        annos_to_process[i]["y_min"]

                if intersection_area(annos_to_process[i], annos_to_process[0]):
                    processing.append((annos_to_process[i], i))

            if len(processing) >= 2:
                ratio_func = lambda x: (x[0]['y_max'] - x[0]['y_min']) ** 0.5 / \
                    (x[0]['x_max'] - x[0]['x_min'])

                processing = sorted(processing, key=ratio_func)

                box = processing[-1]

                new_annotations[path].append(box[0])

                processing = sorted(processing, key=lambda x: -x[1])

            for i in range(len(processing)):
                del annos_to_process[processing[i][1]]

    return new_annotations

def intersection_area(annotation1: Dict[str, int],
                    annotation2: Dict[str, int]) -> int:

    xmin = max(annotation1["x_min"], annotation2["x_min"])
    ymin = max(annotation1["y_min"], annotation2["y_min"])
    xmax = min(annotation1["x_max"], annotation2["x_max"])
    ymax = min(annotation1["y_max"], annotation2["y_max"])

    if annotation1["x_min"] > annotation2["x_max"]:
        return False

    if annotation2["x_min"] > annotation1["x_max"]:
        return False

    if annotation1["y_min"] > annotation2["y_max"]:
        return False

    if annotation2["y_min"] > annotation1["y_max"]:
        return False

    return True