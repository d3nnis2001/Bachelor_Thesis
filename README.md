<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Stargazers][stars-shield]][stars-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/d3nnis2001/Prototype-based-sEMG-signal-classification-with-CNet2D">
    <img src="images/Logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Prototype-based sEMG classification with CNet2D as a Backbone</h3>

  <p align="center">
    This project utilizes prototype-based classification methods such as GLVQ and GMLVQ for movement classification, using CNet2D as a powerful backbone to extract meaningful features. It is compatible with the Ninapro and Nearlab datasets, enabling efficient and accurate sEMG classification.
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#methods">Methods</a>
    </li>
    <li><a href="#experiments">Experimental Results</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![CNet2D][about-pic]]

This project implements PyTorch layers for the machine learning models GLVQ and GMLVQ, along with a CNN architecture (CNet2D) that leverages three convolutional blocks and two dense layers. The network processes features before utilizing the newly implemented layer for classification.

Designed for Few-Shot Learning, the framework allows seamless addition of new prototypes and classes, enabling efficient training on minimal data without the need for full model retraining. The preprocessing pipeline ensures correct loading, filtering, normalization, and segmentation of Nearlab and Ninapro datasets, making the data ready for optimal model performance.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Pytorch][Pytorch]][Pytorch-url]
* [![Pandas][Pandas]][Pandas-url]
* [![scikit-learn][Sklearn]][scikit-learn-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Experimental Results

The experiments evaluate the performance and adaptability of the proposed prototype-based classification layer compared to the traditional Softmax classifier. Key findings include:

[![Accuracy per Participant for each Model][accuracy-pic]]

- **Prototype-based vs. Softmax:** The prototype-based methods (GLVQ, GMLVQ) performed on par with Softmax across different evaluation scenarios, demonstrating comparable accuracy while offering benefits in interpretability and adaptability.
- **Impact of Prototypes per Class:** Increasing the number of prototypes per class slightly improved GLVQ's accuracy, while GMLVQ showed minimal variation, indicating that the CNN backbone already provides strong feature separability.
- **Computational Performance:** Prototype-based layers achieved faster inference times than Softmax, making them more suitable for real-time applications.
- **Few-Shot Learning:** GLVQ outperformed GMLVQ in few-shot learning scenarios, effectively classifying new movement patterns with minimal training samples.

[![Few-shot learning results for GLVQ and GMLVQ][fsl-pic]]

Overall, these results highlight the robustness of prototype-based learning for sEMG classification, particularly in scenarios requiring rapid adaptation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Dennis Schielke - dennis.schielke1@gmail.com

Project Link: [https://github.com/d3nnis2001/Prototype-based-sEMG-signal-classification-with-CNet2D](https://github.com/d3nnis2001/Prototype-based-sEMG-signal-classification-with-CNet2D)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

I would like to express my gratitude to the following:

- **Prof. [Benjamin Paaßen]** for their valuable insights and guidance.
- The **PyTorch**, **Scikit-learn**, and **Pandas** communities for providing excellent open-source tools.
- The **Ninapro** and **Nearlab** teams for making their datasets publicly available.
- The contributors of the **Best-README-Template**, which inspired the structure of this document.
- The author of the **NinaPro-Helper-Library**, which helped preprocessing the Ninapro dataset properly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Links -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/d3nnis2001/Prototype-based-sEMG-signal-classification-with-CNet2D/stargazers
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/dennis-schielke-60b82525a/
[Pytorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[Pytorch-url]: https://pytorch.org/
[Pandas]: https://img.shields.io/badge/-Pandas-150458?&logo=pandas
[Pandas-url]: https://pandas.pydata.org/
[scikit-learn]: https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/

<!-- Images -->
[about-pic]: images/About.png
[fsl-pic]: images/fsl.png
[accuracy-pic]: images/barplot1.png
[logo-pic]: images/Logo.png
