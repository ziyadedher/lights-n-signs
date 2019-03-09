import pickle

DATASET_NAME = "scale_lights"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def preprocess_scale_lights(scale_lights_path: str, proportion: float = 1.0,
                            testset: bool = False) -> Dataset:
    """Preprocess and generate data for our custom dataset at the given path.

    Raises `FileNotFoundError` if any of the required files or folders is not
    found.
    """
    with open(os.join.path(scale_lights_path, 'scale_dataset.pickle'), "rb") as f:
        data = pickle.load(f)

    images: List[str] = data['images']
    detection_classes: List[str] = data['classes']
    annotations: Dict[str, List[Dict[str, int]]] = data['annotations']

    return Dataset(DATASET_NAME, images, classes, annotations)
