import colorlog
import logging

def logging_decorator(project_name):
    def decorator(func):
        # Configure colorlog
        formatter = colorlog.ColoredFormatter(
            "%(asctime)s %(log_color)s[%(levelname)s] \033[34;1m[{}]\033[0m %(message)s".format(project_name),
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

        # Create a logger with colorlog
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        class Wrapper:
            def __call__(self, *args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Log the raise condition with the project name and raised value
                    logger.info("[\033[34;1m{}\033[0m] Raise condition occurred: {}".format(project_name, str(e)))
                    raise

        return Wrapper()

    return decorator
