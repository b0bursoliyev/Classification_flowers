import streamlit as st
from PIL import Image
from fastai.vision import all
import plotly.express as px
import pathlib
import platform

# Adjusting pathlib for Linux

plt = platform.system()
if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

# Web application title
st.title("Flower Classification")
st.info("This model can classify 5 type of flowers. They are daisy,dandelion,roses,sunflowers, and tulips")

# File upload section
uploaded_file = st.file_uploader("Choose an image...", type=[".jpg", ".jpeg", ".png"])

if uploaded_file:
    # Read the image file
    img = Image.open(uploaded_file)

    # Display the image
    st.image(img, caption="Uploaded Image", width=650)

    # Load the pre-trained model
    which = all.load_learner('which.pkl')

    # Predict if image is present
    pred1, pred_id1, probs1 = which.predict(img)

    # Display prediction result
    st.success(f"The predicted: {pred1.title()}")
    st.info(f"Confidence: {100.0 * probs1[pred_id1].item():.2f}%")
    # Bar chart visualization
    chart_data1 = {'Result': which.dls.vocab, 'Probability (%)': probs1 * 100}
    fig1 = px.bar(chart_data1, x='Probability (%)', y='Result', orientation='h',
                 labels={'Probability (%)': 'Probability'})
    st.plotly_chart(fig1)

    #check to flower

    if pred1 == 'flowers':
        #Load img
        model = all.load_learner('flowers.pkl')
        #split items
        pred,pred_id,probs = model.predict(img)
        # Display prediction result
        st.success(f"The flower's type is: {pred.title()}")
        st.info(f"Confidence: {100.0 * probs[pred_id].item():.2f}%")
        # Bar chart visualization
        chart_data = {'Flowers Class': model.dls.vocab, 'Probability (%)': probs * 100}
        fig = px.bar(chart_data, x='Probability (%)', y='Flowers Class',
                     orientation='h',labels={'Probability (%)': 'Probability'})
        st.plotly_chart(fig)
    else:
        st.info("Are you kidding me! Please upload the flower's picture not another one")
    # Close the image file
    img.close()
