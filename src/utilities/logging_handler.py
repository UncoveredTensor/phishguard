import colorlog
import logging

def logging_decorator(project_name: str):
    def decorator(func):
        # Create a logger with colorlog
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()

        class Wrapper:
            def __init__(self):
                self.project_name = project_name

                # Configure colorlog for this function
                formatter = colorlog.ColoredFormatter(
                    "%(asctime)s %(log_color)s[%(levelname)s] \033[34;1m[{}]\033[0m %(message)s".format(self.project_name),
                    log_colors={
                        'DEBUG': 'green',
                        'INFO': 'yellow',
                        'WARNING': 'blue',
                        'ERROR': 'red',
                        'CRITICAL': 'bold_red',
                    },
                    reset=True,
                    style='%'
                )
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

            def __call__(self, *args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Log the raised condition with the project name and raised value
                    logger.info("[\033[34;1m{}\033[0m] Raise condition occurred: {}".format(self.project_name, str(e)))
                    raise

        return Wrapper()

    return decorator
