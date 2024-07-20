"""import fiftyone as fo

export_dir = "/Users/parthbhalodiya/yolo"
label_field = "ground_truth"  # for example

# The splits to export
splits = ["train", "val", "test"]

# All splits must use the same classes list
classes = ["banana", "apple", "sandwich", "orange",  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]


# The dataset or view to export
# We assume the dataset uses sample tags to encode the splits to export
dataset_or_view = fo.Dataset(...)

# Export the splits
for split in splits:
    split_view = dataset_or_view.match_tags(split)
    split_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        split=split,
        classes=classes,
    )
  """""
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["banana", "apple", "sandwich", "orange",  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    max_samples=180,
    shuffle=True,
)