FROM pytorch/pytorch:latest
WORKDIR /studio
COPY ./*.py ./
COPY ./templates/* ./
COPY vqgan_imagenet_f16_1024.yaml .
COPY vqgan_imagenet_f16_1024.ckpt .
COPY ./dockerinit.sh ./
RUN pip install ftfy regex stqdm omegaconf pytorch-lightning einops transformers streamlit ffpb streamlit-ace
RUN apt-get update
RUN apt-get install git npm curl -y
RUN git clone https://github.com/openai/CLIP
RUN git clone --quiet https://github.com/CompVis/taming-transformers
RUN pip install -e ./taming-transformers
RUN npm install -g n
RUN n latest
RUN npm install -g npm
RUN npm install -g localtunnel

# this will be run on vast.ai as a ssh session
# we need to run the init script manually so we can see the url
CMD ["/bin/bash", "dockerinit.sh"]