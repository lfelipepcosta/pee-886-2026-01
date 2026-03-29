__all__ = [ "get_argparser_formatter", "setup_logs"]

import sys
from loguru         import logger
from rich_argparse  import RichHelpFormatter

def get_argparser_formatter():
    RichHelpFormatter.styles["argparse.args"]     = "green"
    RichHelpFormatter.styles["argparse.prog"]     = "bold grey50"
    RichHelpFormatter.styles["argparse.groups"]   = "bold green"
    RichHelpFormatter.styles["argparse.help"]     = "grey50"
    RichHelpFormatter.styles["argparse.metavar"]  = "blue"
    return RichHelpFormatter

def setup_logs( name , level="INFO"):
    """Setup and configure the logger"""
    logger.configure(extra={"name" : name})
    logger.remove()  # Remove any old handler
    #format="<green>{time:DD-MMM-YYYY HH:mm:ss}</green> | <level>{level:^12}</level> | <cyan>{extra[slurms_name]:<30}</cyan> | <blue>{message}</blue>"
    if level=="DEBUG":
        format="<blue>{time:DD-MMM-YYYY HH:mm:ss}</blue> | <level>{level:^12}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> |{message}"
    else:
        format="<blue>{time:DD-MMM-YYYY HH:mm:ss}</blue> | {message}"
    logger.add(
        sys.stdout,
        colorize=True,
        backtrace=True,
        diagnose=True,
        level=level,
        format=format,
    )
    
    

# README.md           clara_pacheco       felipe_grael        guilherme_thomaz    miguel_saavedra
#__init__.py         eduardo_banaczewski felipe_taparo       leandro_fernandes   pedro_achcar
#__pycache__         ellizeu_sena        fernanda_verde      lucas_nunes         pedro_campos
#brenno_rodrigues    eraldo_junior       gabriel_lisboa      luiz_costa          samarone_junior
# include all submodules (c,d,e,...)here...

from . import brenno_rodrigues
__all__.extend( brenno_rodrigues.__all__ )
from .brenno_rodrigues import *

from . import clara_pacheco
__all__.extend( clara_pacheco.__all__ )
from .clara_pacheco import *

from . import eduardo_banaczewski
__all__.extend( eduardo_banaczewski.__all__ )
from .eduardo_banaczewski import *

from . import ellizeu_sena
__all__.extend( ellizeu_sena.__all__ )
from .ellizeu_sena import *

from . import eraldo_junior
__all__.extend( eraldo_junior.__all__ )
from .eraldo_junior import *

from . import felipe_grael
__all__.extend( felipe_grael.__all__ )
from .felipe_grael import *

from . import felipe_taparo
__all__.extend( felipe_taparo.__all__ )
from .felipe_taparo import *

from . import fernanda_verde
__all__.extend( fernanda_verde.__all__ )
from .fernanda_verde import *

from . import gabriel_lisboa
__all__.extend( gabriel_lisboa.__all__ )
from .gabriel_lisboa import *

from . import guilherme_thomaz
__all__.extend( guilherme_thomaz.__all__ )
from .guilherme_thomaz import *

from . import leandro_fernandes
__all__.extend( leandro_fernandes.__all__ )
from .leandro_fernandes import *

from . import lucas_nunes
__all__.extend( lucas_nunes.__all__ )
from .lucas_nunes import *

from . import luiz_costa
__all__.extend( luiz_costa.__all__ )
from .luiz_costa import *

from . import miguel_saavedra
__all__.extend( miguel_saavedra.__all__ )
from .miguel_saavedra import *

from . import pedro_achcar
__all__.extend( pedro_achcar.__all__ )
from .pedro_achcar import *

from . import pedro_campos
__all__.extend( pedro_campos.__all__ )
from .pedro_campos import *

from . import samarone_junior
__all__.extend( samarone_junior.__all__ )
from .samarone_junior import *

from . import group_works
__all__.extend( group_works.__all__ )
from .group_works import *

