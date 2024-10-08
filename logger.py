import logging
logging.basicConfig(filename='logging.log',level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# create a logger
logger=logging.getLogger(__name__)

# log messages
logger.debug('This is a debug message')
logger.info('This is a info message')
logger.warning('This is a warining message')
logger.error('This is a error message')
logger.critical('This is a critical message')
try:
    result=1/0
except Exception as e:
    logger.exception('An exception occurred:%s',e)
    