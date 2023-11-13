# Model Server

The Model Server component uses Nginx to serve JavaScript models for (SSO) web service. It plays a role in providing the necessary models for web frontend.

## Overview

### `model_gens`

This contains code for generating models.

### `based-models`

The `based-models` directory contains models used for web frontend encoding. These models are served to web frontend which is consumed with tensorflowjs GraphModel.

### `descriminator`

The `descriminator` folder stores trained models from the pipeline. These models are then utilized to create the final encoding in the SSO-Server component.


