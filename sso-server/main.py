import uvicorn
import logging

import env

logging.info('Running database migration using alembic')

logging.info(f'Running sso server at port {env.PORT}.')
uvicorn.run("app:ssoApp",
        host='0.0.0.0',
        port=env.PORT,
        log_level='info',
    )