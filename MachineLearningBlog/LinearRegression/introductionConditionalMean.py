import numpy as np
import pandas as pd
import plotly.express as px

# Sample data
np.random.seed(42)
num_guests = np.arange(1, 21)  # Number of guests from 1 to 20
appetite = np.random.uniform(1, 3, size=num_guests.shape)  # Random appetite levels

# Coefficients (magic numbers)
coeff_num_guests = 2  # Coefficient for number of guests
coeff_appetite = 5     # Coefficient for appetite
constant = 3           # Constant (extra)

# Calculate the number of waffles needed using the linear relationship
num_waffles = coeff_num_guests * num_guests + coeff_appetite * appetite + constant

# Create a DataFrame
data = pd.DataFrame({
    'Number of Guests': num_guests,
    'Friends\' Appetite': appetite,
    'Estimated Waffles': num_waffles
})

# Create a 3D scatter plot
fig = px.scatter_3d(data, 
                     x='Number of Guests', 
                     y='Friends\' Appetite', 
                     z='Estimated Waffles',
                     title='Estimating Waffles Based on Guests and Appetite',
                     labels={'Number of Guests': 'Number of Guests',
                             'Friends\' Appetite': 'Friends\' Appetite',
                             'Estimated Waffles': 'Estimated Waffles'},
                     color='Estimated Waffles',
                     size='Estimated Waffles')

# Show the plot
fig.show()
