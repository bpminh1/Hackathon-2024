# Hackathon 2024 - Submission of Group *Miric*

Team members:
    - Phuong Minh Bui
    - Cedric Styra

## Description
We had two different sets of problems we wanted to solve and optimize taking the initial contraints into account: image processing and search.
For image processing we used the OpenCV-Module for Python, to search for 'external contours' marking the edges of the part and search for 'internal contours' marking
holes of the parts we want to avoid. With this processed image we could address the search problem using our search algorithm which minimizes the distance from the part middle and avoids falling into internal contours while staying inside the external contour - the tile. It did this by interatively moving away from the tile middle and
trying out all possible angles 

## How to Run
python solution/main.py solution/input.csv solution/output.csv

## ... and other things you want to tell us
We faced the challenges of having to use machine learning for hole detection... which we didn't. It was fun though!

## License

All resources in this repository are licensed under the MIT License. See the [LICENSE](LICENSE) for more information.

## Acknowledgements

<img src="doc/logos-all.png" alt="Logos" width="600px" />

This project is partially funded by the German Federal Ministry of Education and Research (BMBF) within the “The Future of Value Creation – Research on Production, Services and Work” program (funding number 02L19C150) managed by the Project Management Agency Karlsruhe (PTKA).
The authors are responsible for the content of this publication.
