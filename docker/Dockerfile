FROM public.ecr.aws/docker/library/r-base:4.1.2

ENV TZ=UTC

RUN apt update && apt -y install cmake dirmngr gnupg apt-transport-https ca-certificates software-properties-common \
    libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 \
    libxi6 libxtst6 libcurl4-openssl-dev libv8-dev libxml2-dev \
    wget python3 python3-dev python3-pip python3-venv r-base
RUN pip3 install boto3
RUN R -e "install.packages(c('Robyn', 'reticulate', 'readr', 'jsonlite'))"

RUN mkdir /app
COPY ./install_packages.R /app/install_packages.R
RUN Rscript /app/install_packages.R

ENV PATH="/opt/ml/code:${PATH}"
WORKDIR /opt/ml/code
COPY ./run.sh /opt/ml/code/run.sh
COPY ./entrypoint.R /opt/ml/entrypoint.R
RUN /bin/bash -c 'chmod +x /opt/ml/code/run.sh'
ENTRYPOINT ["/bin/bash", "run.sh"]
