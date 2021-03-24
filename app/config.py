class CommonConfig(object):
    """
    公共配置
    """


class DevelopmentConfig(CommonConfig):
    """
    开发环境配置
    """
    DEBUG = True


class ProductionConfig(CommonConfig):
    """
    正式环境配置
    """
    DEBUG = False


app_config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
}
