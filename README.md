# Single Sign-On (SSO) Service with Face Recognition

Welcome to the Single Sign-On (SSO) service project utilizing face recognition! This project aims to provide user authentication through facial recognition using mobile_net as base model and another model trained using siamese network. The project is organized into several components, each serving a specific purpose.

## Components

- [SSO-Web](./sso-web/README.md)
- [SSO-Server](./sso-server/README.md)
- [Pipeline](./pipeline/README.md)
- [Experiments](./experiments/README.md)

## Dependencies

To run the entire project, follow these steps:

1. Run `docker-compose up` to start the whole project.
2. Perform Alembic migration and Milvus migration to set up the database.
3. In the `sso-server` directory, run `alembic upgrade head` to migrate the PostgreSQL server.
4. Run `setup` from the `repository.milvus` to setup the Milvus database.


