import os
import sys
import socket
import logging
import logging.handlers
from mksc import reader

class Logger(object):
    """
    日志记录类
    """
    def __init__(self):
        self._setup_log_dir()
        self._setup_output_format()
        self._setup_file_handler()
        self._setup_stdout_handler()
        self._setup_loggers()

        self.info = self._logger_info.info
        self.warning = self._logger_warning.warning
        self.error = self._logger_error.error
        self.cfg = reader.config()

    def _setup_log_dir(self):
        """
        设置日志目录
        """
        self._log_dir = os.path.abspath(self.cfg.get('PATH', 'log_dir'))
        if os.path.exists(self._log_dir):
            pass
        else:
            os.mkdir(self._log_dir)
            print('[Logger]: created dir {}'.format(self._log_dir))

    def _setup_output_format(self):
        """
        设置日志格式
        """
        hostname = socket.gethostname()
        header = hostname + '\t' + '%(asctime)s' + '\t' + '%(levelname)s' + '\t' + '%(filename)s:%(lineno)s' + '\n'

        self._info_formatter = logging.Formatter(header + '- [INFO] ' + '%(message)s' + '\n')
        self._warning_formatter = logging.Formatter(header + '- [WARNING]' + '%(message)s' + '\n')
        self._error_formatter = logging.Formatter(header + '- [ERROR]' + '%(message)s' + '\n')

    def _setup_file_handler(self):
        """
        设置日志存储细节
        """
        n_bytes = 1024 * 1024 * 4  # 4MB / file
        n_backups = 10

        # 文件日志
        self._info_file_handler = logging.handlers.RotatingFileHandler(filename=os.path.join(self._log_dir, 'info.log'),
                                                                       mode='a',
                                                                       maxBytes=n_bytes,
                                                                       backupCount=n_backups)
        self._warning_file_handler = logging.handlers.RotatingFileHandler(os.path.join(self._log_dir, 'warning.log'),
                                                                          mode='a',
                                                                          maxBytes=n_bytes,
                                                                          backupCount=n_backups)
        self._error_file_handler = logging.handlers.RotatingFileHandler(os.path.join(self._log_dir, 'error.log'),
                                                                        mode='a',
                                                                        maxBytes=n_bytes,
                                                                        backupCount=n_backups)
        self._info_file_handler.setFormatter(self._info_formatter)
        self._warning_file_handler.setFormatter(self._warning_formatter)
        self._error_file_handler.setFormatter(self._error_formatter)

    def _setup_stdout_handler(self):
        """
        设置标准输出细节
        """
        self._stdout_handler = logging.StreamHandler(sys.stdout)
        self._stderr_handler = logging.StreamHandler(sys.stderr)

    def _setup_loggers(self):
        """
        获取logger实例
        指定日志的最低输出级别
        为logger添加的日志处理器
        """
        self._logger_info = logging.getLogger("info")
        self._logger_info.setLevel(logging.INFO)
        if not self._logger_info.handlers:
            self._logger_info.addHandler(self._stdout_handler)
            self._logger_info.addHandler(self._info_file_handler)

        self._logger_warning = logging.getLogger("warning")
        self._logger_warning.setLevel(logging.WARNING)
        if not self._logger_warning.handlers:
            self._logger_warning.addHandler(self._stdout_handler)
            self._logger_warning.addHandler(self._warning_file_handler)

        self._logger_error = logging.getLogger("error")
        self._logger_error.setLevel(logging.ERROR)
        if not self._logger_error.handlers:
            self._logger_error.addHandler(self._stderr_handler)
            self._logger_error.addHandler(self._error_file_handler)


# logger instance
logger = Logger()

if __name__ == '__main__':
    # test
    logger.info('test logger info')
    logger.warning('test logger warning')
    logger.error('test logger error')
