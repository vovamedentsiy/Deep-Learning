# Deep Learning
Assignments for the Spring 2019 Deep Learning course at University of Amsterdam

## Assignment 1. MLPs, CNNs and Backpropagation
1. Derived analytical gradients for the FC and Batch Normalization layers
2. Numpy [implementation](https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment1/code/modules.py) of the FC and activation functions with custom backward methods 
3. Pytorch [implementation](https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment1/code/custom_batchnorm.py) of the BatchNorm module with custom backward method
4. Trained MLP and simple CNN on the Cifar10 dataset

<img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment1/imgs/accuracy_np.jpg" width="250" height ="200" /> <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment1/imgs/accuracy_mlp.jpg" width="250" height ="200" /> <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment1/imgs/accuracy_cnn.jpg" width="250" height ="200" />

Find report with derivations [here ](https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment1/DL_assignment1.pdf)


Find code here [here ](https://github.com/vovamedentsiy/Deep-Learning/tree/main/medentsiy_assignment1/code)


## Assignment 2: Recurrent Neural Networks and Graph Neural Networks

1. Derived analytical gradients for the RNN
2. Pytorch implementation of the [vanilla RNN](https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment2/code/part1/vanilla_rnn.py) and [LSTM](https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment2/code/part1/lstm.py) trained to memorize palindromes with various length
<p> 
    <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment2/imgs/rnn.jpg" width="300" height ="200" /> <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment2/imgs/lstm.jpg" width="300" height ="200" /> 
    <br>
<p\>
    
3. Trained [LSTM](https://github.com/vovamedentsiy/Deep-Learning/tree/main/medentsiy_assignment2/code/part2) as a generative model to predict the next character in the text


Find report with derivations [here ](https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment2/DL_assignment2.pdf)


Find code here [here ](https://github.com/vovamedentsiy/Deep-Learning/tree/main/medentsiy_assignment2/code)

## Assignment 3: Deep Generative Models
1. Pytorch [implementation](https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/code/a3_vae_template.py) of VAE
<p> 
    <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/vae/elbo.png" width="400" height ="250" />
    <br>
    <em>ELBO for VAE with 20-dim latent space</em> 
<p\>
  
<p> 
    <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/vae/sampled0.png" width="270" height ="235" /> <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/vae/sampled20.png" width="270" height ="235" /> <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/vae/sampled40.png" width="270" height ="235" />
    <br>
    <em>Images sampled from Decoder at the begining, halfway through, and at the end of training </em> 
<p\>

<p> 
    <img align="center" src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/vae/manifold.png" width="480" height ="400" />
    <br>
    <em>Output of VAE’s decoder in 2-dimensional latent space</em> 
<p\>
    
2. Pytorch [implementation](https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/code/a3_gan_template.py) of GAN
<p> 
    <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/gan/loss_gan.png" width="400" height ="250" />
    <br>
    <em>Loss of Generator and Discriminator networks</em> 
<p\>
  
<p> 
    <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/gan/0.png" width="300" height ="250" /> <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/gan/gan_500.png" width="300" height ="250" /> <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/gan/gan_100000.png" width="300" height ="250" /> <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/gan/186500.png" width="300" height ="250" />
    <br>
    <em>Images sampled from Generator at the begining, halfway through, and at the end of training</em> 
<p\>

<p> 
    <img align="center" src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/gan/generated_transform.png" width="400" height ="70" />
    <br>
    <em>Interpolating between two images in the latent space</em> 
<p\>
    
    
3. Pytorch [implementation](https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/code/a3_nf_template.py) of RealNVP
<p> 
    <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/nf/avg_bpd.png" width="400" height ="250" />
    <br>
    <em>Training and validation performance in bits per dimension</em> 
<p\>
  
<p> 
    <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/nf/sampled_img_nvp0.png" width="290" height ="235" /> <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/nf/sampled_img_nvp19.png" width="290" height ="235" /> <img src="https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/imgs/nf/sampled_img_nvp39.png" width="290" height ="235" />
    <br>
    <em>Images sampled from RealNVP after the 1-st epoch, at the middle point of training, and at the end of training </em> 
<p\>

Find report [here ](https://github.com/vovamedentsiy/Deep-Learning/blob/main/medentsiy_assignment3/DL_assignment3.pdf)


Find code here [here ](https://github.com/vovamedentsiy/Deep-Learning/tree/main/medentsiy_assignment3/code)
