"""
Implements logging.
"""

from neural_network import utility

NONE = 0
ADVANCED = 1
BASIC = 2
SUPERFICIAL = 3
ALL = 4

LOG_STRINGS = {}

__all__ = ["log",
           "LOG_STRINGS",
           "NONE",
           "ADVANCED",
           "BASIC",
           "SUPERFICIAL",
           "ALL"]


def _append_log_(message,
                 *args,
                 filename='train.log',
                 lock=None):
    if lock:
        lock.acquire()
    log_file = open(filename, mode='a')
    print(LOG_STRINGS["Header"][1].format(utility.timestamp()), file=log_file, end='')
    print(message.format(*args), file=log_file, end='')
    print(LOG_STRINGS["Footer"][1].format(), file=log_file, end='')
    log_file.close()
    if lock:
        lock.release()


def _create_logstring_(log_name,
                       param_tuple,
                       message):
    if log_name in LOG_STRINGS:
        return False

    LOG_STRINGS[log_name] = \
        (param_tuple, message)
    return True


def log(log_string,
        *args,
        filename='train.log',
        lock=None):
    """
    Logs to :filename some message from LOG_STRINGS.
    """
    if log_string not in LOG_STRINGS:
        raise ValueError("Log string not found or initialized")
    log_types, message = LOG_STRINGS[log_string]
    comparison = zip(args, log_types)
    test = [issubclass(type(obj), required_type) for obj, required_type in comparison]
    if not all(test):
        error_msg = """
Some of position arguments were of wrong type.
Message: {0}
Args len: {1}
Args:
""".format(log_string, len(args))
        error_msg = (error_msg+error_msg.join(str(this_str)+'\n' for this_str in args))
        raise TypeError(error_msg)
    _append_log_(message, *args, filename=filename, lock=lock)


def _init_strings_():
    _create_logstring_("Header",
                       ((int, float),),
                       """
{0}:
-----------------------------------------------------------------------------------
                       """)

    _create_logstring_("Footer",
                       (float,),
                       """
-----------------------------------------------------------------------------------
                       """)

    _create_logstring_("Random Training Start",
                       ((int, float),
                        (int, float),
                        (int, float),
                        (int, float),
                        (int, float),
                        int,
                        str,
                        str),
                       """
Random training has been started.
The parameters are as follows:
Bulk report limit: {0}
Delta: {1}
Multiplier: {2}
Satisfactory limit: {3}
Diminishing return limit: {4}
Guaranteed epochs: {5}
Intermediate results are saved to {6}
Best network will be saved to {7}
                       """)

    _create_logstring_("Gradient Training Start",
                       (int,
                        (int, float),
                        (int, float),
                        (int, float),
                        (int, float),
                        (int, float),
                        str,
                        str),
                       """
Gradient Descent training has been started.
The parameters are as follows:
Epochs: {0}
Delta: {1}
Multiplier: {2}
Learning Rate: {3}
Satisfactory limit: {4}
Diminishing return limit: {5}
Intermediate results are saved to {6}
Best network will be saved to {7}
                       """)

    _create_logstring_("Gradient Descent Fail",
                       ((int, float),),
                       """
Gradient Descent failed to decrease the objective function after
  finding the derivatives for this network.
This might mean that the limit of this neural network has been reached
  or the parameters are too high.

Objective function is {0}
                       """)

    _create_logstring_("Satisfactory Limit Reached",
                       ((int, float),),
                       """
Satisfactory limit was reached.
Objective function is {0}
No further training is required.
                       """)

    _create_logstring_("Epoch Limit Reached",
                       ((int, float),),
                       """
Epoch limit was reached.
Satisfactory error couldn't be achived, further training is required.
Objective function is {0}
                       """)

    _create_logstring_("Attempt Limit Reached",
                       ((int, float),),
                       """
Attempt limit was reached.
Further training is required, lower delta or use better training method.
Objective function is {0}
                       """)

    _create_logstring_("Progress Saved",
                       (str,),
                       """
The network was saved to {0}
                       """)

    _create_logstring_("Epoch Passed",
                       (int,
                        (int, float),
                        (int, float)),
                       """
Epoch has been passed.
Current epoch: {0}
Current objective function: {1}
Delta-loss of objective function is {2:%}
                       """)

    _create_logstring_("Derivation Time",
                       (int,
                        (int, float)),
                       """
Calculation of derivatives was started.

Derivatives to be found: {0}
It will take approximately: {1:3} s
                       """)

    _create_logstring_("Diminishing Return",
                       ((int, float),),
                       """
Seems like further training is pointless as last training session brought
  barely noticeable objective function's change.
Delta-loss: {0:%}
                       """)

    _create_logstring_("Life Log",
                       ((int, float),
                        (int, float),
                        (int,),
                        (int,)),
                       """
Generation {3} has just finished evolving.
Previous fitness: {0}
New fitness: {1}
Networks to be trained: {2}
                       """)

_init_strings_()
