from fastbook import *
from fastai.vision.widgets import *
from fastai.imports import *
import gradio as gr
from os import listdir

learn2 = load_learner("mymodel.pkl")

cateogries = ('John', 'Cat', 'Dog', 'Human')
examples = [f'example_images/{x}' for x in listdir('example_images')]
def classify_image(img):
  pred, idx, probs = learn2.predict(img)
  return dict(zip(cateogries, map(float, probs)))

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)