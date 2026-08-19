# DT-909

DT-909 is a tool for automatically creating captions for character image datasets.

Unlike normal batch captioning, it looks at the dataset as a whole instead of treating every image as completely separate. This helps keep descriptions more consistent across different poses, lighting conditions, crops, outfits, and camera angles.

### The idea

**Observe each image → compare the dataset → create the captions**

DT-909 is designed to reduce repetitive captioning work while still keeping the final dataset easy to review and correct.

## Usage

```bash
python cli.py "/path/to/dataset"
```

Captions are saved as `.txt` files next to the corresponding images.

```text
dataset/
├── image_001.png
├── image_001.txt
├── image_002.png
└── image_002.txt
```

# Contributing
Contributions are welcome! Please fork the repository and submit pull requests.

# License
This project is licensed under the MIT License.

# Acknowledgements
Martin Bosgra: Author and primary maintainer of the project.
