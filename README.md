# Semantic Segmentation on MIT ADE20K dataset in PyTorch

This is a PyTorch implementation of semantic segmentation models on MIT ADE20K scene parsing dataset (http://sceneparsing.csail.mit.edu/).


Color encoding of semantic categories can be found here:
https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit?usp=sharing



Encoder:
- MobileNetV2dilated
- ResNet18/ResNet18dilated
- ResNet50/ResNet50dilated
- ResNet101/ResNet101dilated
- HRNetV2 (W48)

Decoder:
- C1 (one convolution module)
- C1_deepsup (C1 + deep supervision trick)
- PPM (Pyramid Pooling Module, see [PSPNet](https://hszhao.github.io/projects/pspnet) paper for details.)
- PPM_deepsup (PPM + deep supervision trick)
- UPerNet (Pyramid Pooling + FPN head, see [UperNet](https://arxiv.org/abs/1807.10221) for details.)

## Performance:
IMPORTANT: The base ResNet in our repository is a customized (different from the one in torchvision). The base models will be automatically downloaded when needed.

<table><tbody>
    <th valign="bottom">Architecture</th>
    <th valign="bottom">MultiScale Testing</th>
    <th valign="bottom">Mean IoU</th>
    <th valign="bottom">Pixel Accuracy(%)</th>
    <th valign="bottom">Overall Score</th>
    <th valign="bottom">Inference Speed(fps)</th>
    <tr>
        <td rowspan="2">MobileNetV2dilated + C1_deepsup</td>
        <td>No</td><td>34.84</td><td>75.75</td><td>54.07</td>
        <td>17.2</td>
    </tr>
    <tr>
        <td>Yes</td><td>33.84</td><td>76.80</td><td>55.32</td>
        <td>10.3</td>
    </tr>
    <tr>
        <td rowspan="2">MobileNetV2dilated + PPM_deepsup</td>
        <td>No</td><td>35.76</td><td>77.77</td><td>56.27</td>
        <td>14.9</td>
    </tr>
    <tr>
        <td>Yes</td><td>36.28</td><td>78.26</td><td>57.27</td>
        <td>6.7</td>
    </tr>
    <tr>
        <td rowspan="2">ResNet18dilated + C1_deepsup</td>
        <td>No</td><td>33.82</td><td>76.05</td><td>54.94</td>
        <td>13.9</td>
    </tr>
    <tr>
        <td>Yes</td><td>35.34</td><td>77.41</td><td>56.38</td>
        <td>5.8</td>
    </tr>
    <tr>
        <td rowspan="2">ResNet18dilated + PPM_deepsup</td>
        <td>No</td><td>38.00</td><td>78.64</td><td>58.32</td>
        <td>11.7</td>
    </tr>
    <tr>
        <td>Yes</td><td>38.81</td><td>79.29</td><td>59.05</td>
        <td>4.2</td>
    </tr>
    <tr>
        <td rowspan="2">ResNet50dilated + PPM_deepsup</td>
        <td>No</td><td>41.26</td><td>79.73</td><td>60.50</td>
        <td>8.3</td>
    </tr>
    <tr>
        <td>Yes</td><td>42.14</td><td>80.13</td><td>61.14</td>
        <td>2.6</td>
    </tr>
    <tr>
        <td rowspan="2">ResNet101dilated + PPM_deepsup</td>
        <td>No</td><td>42.19</td><td>80.59</td><td>61.39</td>
        <td>6.8</td>
    </tr>
    <tr>
        <td>Yes</td><td>42.53</td><td>80.91</td><td>61.72</td>
        <td>2.0</td>
    </tr>
    <tr>
        <td rowspan="2">UperNet50</td>
        <td>No</td><td>40.44</td><td>79.80</td><td>60.12</td>
        <td>8.4</td>
    </tr>
    <tr>
        <td>Yes</td><td>41.55</td><td>80.23</td><td>60.89</td>
        <td>2.9</td>
    </tr>
    <tr>
        <td rowspan="2">UperNet101</td>
        <td>No</td><td>42.00</td><td>80.79</td><td>61.40</td>
        <td>7.8</td>
    </tr>
    <tr>
        <td>Yes</td><td>42.66</td><td>81.01</td><td>61.84</td>
        <td>2.3</td>
    </tr>
    <tr>
        <td rowspan="2">HRNetV2</td>
        <td>No</td><td>42.03</td><td>80.77</td><td>61.40</td>
        <td>5.8</td>
    </tr>
    <tr>
        <td>Yes</td><td>43.20</td><td>81.47</td><td>62.34</td>
        <td>1.9</td>
    </tr>

</tbody></table>

