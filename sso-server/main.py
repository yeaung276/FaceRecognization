import uvicorn
import logging

import env

if __name__=='__main__':
    logging.info(f'Running sso server at port {env.PORT}.')
    uvicorn.run("app:ssoApp",
            host='0.0.0.0',
            port=env.PORT,
            log_level='info',
            reload=True
        )