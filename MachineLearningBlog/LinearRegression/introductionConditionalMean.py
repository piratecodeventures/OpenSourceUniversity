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
    'Estimated Waffles': num_waffles
})

# Create a line plot
fig = px.line(data, 
               x='Number of Guests', 
               y='Estimated Waffles',
               title='Estimated Waffles Based on Number of Guests',
               labels={'Number of Guests': 'Number of Guests',
                       'Estimated Waffles': 'Estimated Waffles'})

# Add scatter points for individual appetite effects
fig.add_scatter(x=num_guests, y=num_waffles, mode='markers', name='Appetite Impact', marker=dict(color='red'))

# Save the plot as a PNG file
fig.write_image("estimated_waffles.png")

# Show the plot
fig.show()
