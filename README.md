# Vanilla VAE Implementation

Make sure you have all the packages in `requirements.txt` installed. You may use the
```pip3 install -r requirements.txt ```

After installation, run the `run_vae.py` file. Inside the file you can find the necessary configuration parameters for the VAE. Setting the training flag to 0, will trigger the evaluation and a plot of 200 samples is saved in the project directory.
If the latent (z) dimension is 2, in addition, you can find a plot of the distributions of the real digits in the latent space.

If you decide to trainn your own VAE, it should only take about 5min.
