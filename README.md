repo
<!-----

You have some errors, warnings, or alerts. If you are using reckless mode, turn it off to see inline alerts.
* ERRORs: 0
* WARNINGs: 0
* ALERTS: 5

Conversion time: 1.336 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β33
* Thu Apr 14 2022 04:51:14 GMT-0700 (PDT)
* Source doc: Reproducibility Blog
* Tables are currently converted to HTML tables.
* This document has images: check for >>>>>  gd2md-html alert:  inline image link in generated source and store images to your server. NOTE: Images in exported zip file from Google Docs may not appear in  the same order as they do in your doc. Please check the images!


WARNING:
You have 8 H1 headings. You may want to use the "H1 -> H2" option to demote all headings by one level.

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 1; ALERTS: 5.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>
<a href="#gdcalert2">alert2</a>
<a href="#gdcalert3">alert3</a>
<a href="#gdcalert4">alert4</a>
<a href="#gdcalert5">alert5</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



```


# Reproduction of A CFCC-LSTM Model for Sea Surface Temperature Prediction


## Reproducibility project - CS4240 Deep Learning


## Group 86 - Matthijs Arnoldus (@: m.arnoldus-1@student.tudelft.nl, 4928091), Marius Birkhoff (@: M.J.H.Birkhoff@student.tudelft.nl, 4724259), Luuk Haarman (@: L.A.Haarman@student.tudelft.nl, 4931173)

Reproducibility is an important aspect of scientific research. This blog describes the process of group 86 for TU Delft's Deep Learning course's efforts to reproduce the results of Yang et al's "A CFCC-LSTM Model for Sea Surface Temperature Prediction". A combined fully connected long short-term memory convolutional neural network for predicting the sea surface temperature was presented in their paper. The efficiency of their model was validated using data from the Bohai Sea in China and the China Ocean. 
The goal of this project is to reproduce and verify the results of Table 1 and Figure 2 of Yung et al.'s paper. For these figures, the Bohai data set was used. 


# Data collection
The next large step to take was to get our hands on the correct data. Because Yung et al. did not specify where their data came from, we had a hard time finding a data set that would satisfy our needs. With some help from our TA and external supervisor, our eyes fell on the Physical Sciences Laboratory by the National Oceanic and Atmospheric Administration (PSL NOAA), which has over 40 years of worldwide data on sea surface temperature. Further exploration of the PSL NOAA lead to yearly datasets. Downloading them from 1981 up until 2021 resulted in 41 files of almost 18 Gigabytes in total.
Only a small portion of the retrieved data was required because we were evaluating using the Bohai Sea and not using the entire planet. This data transformation could be executed quite easily and would save a lot of storage and memory. By making use of the command line interface Climate Data Operators (CDO), we could merge all points in time into one file. Once done, the latitude and longitude dimensions could be reduced to only the Bohai Sea. The result set was 10 megabytes in size and contained 14732 points in time. The following commands were executed:


    cdo -mergetime ./data/*.nc ./sst.worldwide.day.mean.1981-2021.nc
    cdo -sellonlatbox,117,122,37,42 ./sst.worldwide.day.mean.1981-2021.nc ./sst.bohai.day.mean.1981-2021.nc




# Model description
The paper proposes three models used for sea surface temperature prediction. The first model is an FC-LSTM model. This model takes a grid as input and returns a grid as output as well. This part of the model is used for learning the temporal relationships (the effect of previous days on our prediction). The other two models are CFCC-LSTM, FC-LSTM models with an added convolutional layer. These convolutional layers are used for learning the spatial relationships (the effect of nearby temperature points on a center point). The difference between the two CFCC-LSTM models is that one uses average convolution, and the other uses weights that it can learn to optimize the prediction.
```




<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")
[^1]


```
Figure from the paper that describes the pre-processing step as well as the model architecture.


# Implementation
For the implementation of the model we used PyTorch, and the model was implemented in Google Colab. Our first steps were to perform the preprocessing of the data. This included the splitting of the data into three distinct lists. A list of latitude values, a list of longitude values, and a list of the SST (Sea surface temperature) values. The SST list used with latitude and longitude coordinates could be used to retrieve the temperature value of that specific point on a specific day. Because land temperatures are stored as extremely large negative values, we clip the temperatures in the SST list as between 0 and infinity. If we do not clip the temperatures the initial prediction errors will be too large for learning. 

After this, some additional steps are required before the data is used for training. These steps include splitting the data into a training and validation set. We used an 80/20 split, so 80% of the data was used for training, and 20% for validation. When preparing the data for training, we used a sliding window with the length of our history length for the input values, and a sliding window with the length of our prediction length for the target values. So if we want to use 4 days to predict the 5th, we use a sliding window size of 4 and 1 for training and target values respectively. After this, some transformations were applied to these samples to fit them to the model, namely flattening the grid of 5x5 values to a list of 25 values such that the LSTM layer could take them as input. 

The FC-LSTM model was implemented as follows: a single LSTM layer would take the input and return an output with the same dimensions, this output would be fed into a Linear layer which would combine the history into a new grid with the number of days we want to predict. After this, a ReLU transformation is done and the data is transformed into a 5x5 grid again. In the CFCC-LSTM Model, this transformation would not occur, but the output of the FC-LSTM model would be fed into a convolutional layer to predict the temperature of a single point instead of a grid. 

During the training process, we used root mean squared error loss and an Adam optimizer with a learning rate of lr=0.001. 

Below is our FC-LSTM class and CFCC-LSTM class. The CFCC-LSTM class extends the FC-LSTM class.



```
class FCLSTM(nn.Module):
  def __init__(self, input_size=25, hidden_size=25, history_length=4, prediction_length=1, device='cpu'): 
    super(FCLSTM, self).__init__()
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    
    self.history_length = history_length
    self.prediction_length = prediction_length

    self.device = device

    self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    self.fc_layer = nn.Linear(history_length * hidden_size, prediction_length * hidden_size)
    self.relu = nn.ReLU()

  # takes in a grid of N x history_length x 25 and returns a grid of N x prediction_length x 5 x 5 
  def forward(self, input, reshape=True):
      num_samples = input.shape[0]
      h_t = torch.zeros(1, num_samples, self.hidden_size).to(self.device)
      c_t = torch.zeros(1, num_samples, self.hidden_size).to(self.device)

      output, (h_t, c_t) = self.lstm_layer(input, (h_t, c_t)) # output: N x history_length x 25
      output = torch.flatten(output, start_dim=1) # = N x history_length * 25
      output = self.fc_layer(output) # = N x prediction_length * 25
      if reshape:
        output = torch.reshape(output, (num_samples, self.prediction_length, int(np.sqrt(self.hidden_size)), int(np.sqrt(self.hidden_size)))) # N x prediction_length x 5 x 5

      output = self.relu(output)
      return output
```



```
class CFCCLSTM(FCLSTM):
  def __init__(self, input_size=25, hidden_size=25, history_length=4, prediction_length=1, device="cpu", mode="weighted"): 
    super(CFCCLSTM, self).__init__(input_size, hidden_size, history_length, prediction_length, device)

    self.conv_layer = nn.Conv1d(1, 1, kernel_size=hidden_size, stride=1, padding=0)
    if mode == "average":
      self.conv_layer = nn.Conv1d(1,1, kernel_size=hidden_size, stride=1, padding=0)
      with torch.no_grad():
        self.conv_layer.weight.data = torch.mul(torch.ones(1, 1, hidden_size), 1.0 / hidden_size)
        self.conv_layer.weight.requires_grad = False

  # takes in a grid of N x history_length x 25 and returns a grid of N x prediction_length x 5 x 5 
  def forward(self, input):
      output = super().forward(input, False) # N x prediction_length x 5 x 5

      # put data in 1 channel
      output = torch.unsqueeze(output, dim=1) # N x 1 x 25
      output = self.conv_layer(output) # N x 1 x 1
      output = torch.flatten(output, start_dim=1)

      return output
```



```


# Hyperparameter tuning

The paper provides a few learning rate options. For us, these were much too high. This could be due to differences in our model architecture. Due to time constraints, hyperparameter tuning was mostly done only for the FC-LSTM model. After some tuning, a learning rate of lr=0.001 seemed to perform well. This learning rate was also used for the CFCC-LSTM Models and seemed to perform well as well. We randomly initialized our weights, similar to the paper. The paper states that they achieved optimal performance with a fixed initialization scheme but does not provide these values.

We used the same number of epochs as the paper (50 and 100), but due to the differences in our model, we are not quite sure whether or not these numbers are optimal for our model. If we had had more time we would have run the model for longer iterations to see if a large jump in performance was possible. When we looked at the gradients of our loss this is likely not the case, but this could have been due to local minima and a larger number of epochs could have possibly helped escape these local minima. 


# Results
Below are given Table I and Figure 2 from the paper. Below those are our reproductions of the table and figure. 
```




<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image2.png "image_tooltip")



```
Table I in the paper
```




<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image3.png "image_tooltip")


<p id="gdcalert4" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert5">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image4.png "image_tooltip")



```
Table I results for our reproduction
```




<p id="gdcalert5" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image5.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert6">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image5.png "image_tooltip")



```
Figure 2 recreated using our model and the Bohai Sea. The first row shows our predicted values, the second the ground truth values, and the third the absolute differences. 

As can be seen in our table, our RMSE scores are similar to those of the paper, albeit a bit higher. This could be due to the differences in our model. The trends seem similar, however, our average convolution outperforms the weighted convolution slightly. This could be due to initialization or that longer training would result in weighted outperforming since weighted should be able to learn average convolution as well. 

For Figure 2 we used the Bohai Sea instead of the China Ocean since we only used this data. 

Our accuracy values are much different, and not necessarily between 0 and 100. This is because the accuracy formula provided does not bound the values, and will only be bounded when the absolute error is very small, and smaller than the original temperature predicted. In other words, the accuracy metric seems to us like a scaled-down version of mean absolute error and could be negative when the absolute error is large and does not necessarily provide an insight into prediction accuracy that RMSE does not.


# Reproduction hurdles
During the reproduction we encountered several difficulties, which could have been prevented by better documentation of methods and tools used in the original paper. We will describe what problems we encountered and how we have overcome these issues in our reproduction.

First of all, the dataset used was not provided in the paper. Because of this, we had to find a dataset of the sea temperatures in the Bohai Sea ourselves. Therefore, we cannot be sure whether or not the data we used reflects the data used for the original paper. 

Because of some information missing and possible ambiguities in the paper, we had to make some implementation decisions that we were not sure properly reflected the original model. For example, the usage of the word layer in the description of the FC-LSTM model could describe a single layer, such as a single LSTM layer, or rather that it's a part of the complete FC-LSTM model. We made the decision to use a single layer for this. Because of this, we have a single LSTM layer, which feeds into a single Linear layer which feeds into a single Conv1d Layer. We are not sure whether this was what the paper describes, or whether this solution performs as well as a model with multiple layers could. We see this as a possible reason for our model having a higher RMSE, and also why the provided learning rates are much different than the learning rates that performed well for our model. The paper does not describe any non-linear activations, although this is quite common in model architectures and including it in ours proved beneficial. Their training and validation details are not provided, so we went with an 80/20 split but we are not sure whether or not they did this or even used a validation set.  

Some other missing implementation details could have had an effect as well. We used PyTorch, but we are not sure what language or tool was used by the original paper, this could have had minimal effects on the results. 



# Conclusion
To conclude, we were somewhat able to reproduce the results that we set out to, although our RMSE scores were somewhat higher and we argue that the accuracy metric is not much different from RMSE in the information it provides. The differences in score could be due to our implementation being different from the original in model architecture and hyperparameters. These differences occurred because we had to make some of our own implementation choices due to information missing.



# Individual Contributions
Matthijs Arnoldus 	Model construction, model evaluation
Marius Birkhoff 		Data gathering, data preprocessing, code refactoring
Luuk Haarman		Model construction, model evaluation

Acknowledgements
NOAA High Resolution SST data provided by the NOAA/OAR/ESRL PSL, Boulder, Colorado, USA, from their Web site
Y. Yang, J. Dong, X. Sun, E. Lima, Q. Mu and X. Wang, "A CFCC-LSTM Model for Sea Surface Temperature Prediction," in IEEE Geoscience and Remote Sensing Letters, vol. 15, no. 2, pp. 207-211, Feb. 2018, doi: 10.1109/LGRS.2017.2780843.


<!-- Footnotes themselves at the bottom. -->
## Notes

[^1]:
     Sourced from Yang et al. (2018) “A CFCC-LSTM Model for Sea Surface Temperature Prediction”
