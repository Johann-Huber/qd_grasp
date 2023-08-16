# taken from https://github.com/ros/xacro/blob/dashing-devel/xacro/__init__.py
# without ros dependencies and command line option
# therefore, do not use $(find <pkg>), but relative path instead when including a xacro file

import ast
import glob
import math
import os
import re
import sys
import xml.dom.minidom
from io import StringIO

from copy import deepcopy
import gym_envs.envs.src.env_constants as env_consts

# colors ##########################
# bold colors
_ansi = {'red': 91, 'yellow': 93}


def is_tty(stream):  # taken from catkin_tools/common.py
    """Returns True if the given stream is a tty, else False"""
    return hasattr(stream, 'isatty') and stream.isatty()


def colorize(msg, color, file=sys.stderr, alt_text=None):
    if color and is_tty(file):
        return '\033[%dm%s\033[0m' % (_ansi[color], msg)
    elif alt_text:
        return '%s%s' % (alt_text, msg)
    else:
        return msg


def message(msg, *args, **kwargs):
    file = kwargs.get('file', sys.stderr)
    alt_text = kwargs.get('alt_text', None)
    color = kwargs.get('color', None)
    print(colorize(msg, color, file, alt_text), *args, file=file)


def warning(*args, **kwargs):
    defaults = dict(file=sys.stderr, alt_text='warning: ', color='yellow')
    defaults.update(kwargs)
    message(*args, **defaults)


def error(*args, **kwargs):
    defaults = dict(file=sys.stderr, alt_text='error: ', color='red')
    defaults.update(kwargs)
    message(*args, **defaults)

# xmlutils ###########################
#from xmlutils import opt_attrs, reqd_attrs, first_child_element, next_sibling_element, replace_node
import xml.dom.minidom


def first_child_element(elt):
    c = elt.firstChild
    while c and c.nodeType != xml.dom.Node.ELEMENT_NODE:
        c = c.nextSibling
    return c


def next_sibling_element(node):
    c = node.nextSibling
    while c and c.nodeType != xml.dom.Node.ELEMENT_NODE:
        c = c.nextSibling
    return c


def replace_node(node, by, content_only=False):
    parent = node.parentNode

    if by is not None:
        if not isinstance(by, list):
            by = [by]

        # insert new content before node
        for doc in by:
            if content_only:
                c = doc.firstChild
                while c:
                    n = c.nextSibling
                    parent.insertBefore(c, node)
                    c = n
            else:
                parent.insertBefore(doc, node)

    # remove node
    parent.removeChild(node)


def attribute(tag, a):
    """
    Helper function to fetch a single attribute value from tag
    :param tag (xml.dom.Element): DOM element node
    :param a (str): attribute name
    :return: attribute value if present, otherwise None
    """
    if tag.hasAttribute(a):
        # getAttribute returns empty string for non-existent attributes,
        # which makes it impossible to distinguish with empty values
        return tag.getAttribute(a)
    else:
        return None


def opt_attrs(tag, attrs):
    """
    Helper routine for fetching optional tag attributes
    :param tag (xml.dom.Element): DOM element node
    :param attrs [str]: list of attributes to fetch
    """
    return [attribute(tag, a) for a in attrs]


def reqd_attrs(tag, attrs):
    """
    Helper routine for fetching required tag attributes
    :param tag (xml.dom.Element): DOM element node
    :param attrs [str]: list of attributes to fetch
    :raise RuntimeError: if required attribute is missing
    """
    result = opt_attrs(tag, attrs)
    for (res, name) in zip(result, attrs):
        if res is None:
            raise RuntimeError("%s: missing attribute '%s'" % (tag.nodeName, name))
    return result


# Better pretty printing of xml
# Taken from http://ronrothman.com/public/leftbraned/xml-dom-minidom-toprettyxml-and-silly-whitespace/
def fixed_writexml(self, writer, indent="", addindent="", newl=""):
    # indent = current indentation
    # addindent = indentation to add to higher levels
    # newl = newline string
    writer.write(indent + "<" + self.tagName)

    attrs = self._get_attributes()
    a_names = sorted(attrs.keys())

    for a_name in a_names:
        writer.write(' %s="' % a_name)
        xml.dom.minidom._write_data(writer, attrs[a_name].value)
        writer.write('"')
    if self.childNodes:
        if len(self.childNodes) == 1 \
           and self.childNodes[0].nodeType == xml.dom.minidom.Node.TEXT_NODE:
            writer.write(">")
            self.childNodes[0].writexml(writer, "", "", "")
            writer.write("</%s>%s" % (self.tagName, newl))
            return
        writer.write(">%s" % newl)
        for node in self.childNodes:
            # skip whitespace-only text nodes
            if node.nodeType == xml.dom.minidom.Node.TEXT_NODE and \
                    (not node.data or node.data.isspace()):
                continue
            node.writexml(writer, indent + addindent, addindent, newl)
        writer.write("%s</%s>%s" % (indent, self.tagName, newl))
    else:
        writer.write("/>%s" % newl)


# replace minidom's function with ours
xml.dom.minidom.Element.writexml = fixed_writexml


# substituition_args #################

class SubstitutionException(Exception):
    """Base class for exceptions in substitution_args routines."""
    pass


class ArgException(SubstitutionException):
    """Exception for missing $(arg) values."""
    pass


def _eval_env(name):
    """
    Returns the environment variable value or throws exception.

    @return: enviroment variable value
    @raise SubstitutionException: if environment variable not set
    """
    try:
        return os.environ[name]
    except KeyError as e:
        raise SubstitutionException(
            'environment variable %s is not set' % str(e))


def _env(resolved, a, args, context):
    """
    Process $(env) arg.

    @return: updated resolved argument
    @rtype: str
    @raise SubstitutionException: if arg invalidly specified
    """
    if len(args) != 1:
        raise SubstitutionException(
            '$(env var) command only accepts one argument [%s]' % a)
    return resolved.replace('$(%s)' % a, _eval_env(args[0]))


def _eval_optenv(name, default=''):
    """
    Eval_optenv

    Returns the value of the environment variable or default

    @name: name of the environment variable
    @return: enviroment variable value or default
    """
    if name in os.environ:
        return os.environ[name]
    return default


def _optenv(resolved, a, args, context):
    """
    Process $(optenv) arg.

    @return: updated resolved argument
    @rtype: str
    @raise SubstitutionException: if arg invalidly specified
    """
    if len(args) == 0:
        raise SubstitutionException(
            '$(optenv var) must specify an environment variable [%s]' % a)
    return resolved.replace('$(%s)' % a, _eval_optenv(args[0], default=' '.join(args[1:])))


def _eval_dirname(filename):
    """
    Gets the absolute path of a given filename

    @param filename
    @return: absolute path
    @rtype path
    """
    if not filename:
        raise SubstitutionException('Cannot substitute $(dirname),'
                                    'no file/directory information available.')
    return os.path.abspath(os.path.dirname(filename))


def _dirname(resolved, a, args, context):
    """
    Process $(dirname).

    @return: updated resolved argument
    @rtype: str
    @raise SubstitutionException: if no information about the current launch file is available,
    for example if XML was passed via stdin, or this is a remote launch.
    """
    return resolved.replace('$(%s)' % a, _eval_dirname(context.get('filename', None)))


def _eval_find(pkg):
    return get_package_share_directory(pkg)

import gym_envs
from pathlib import Path
def _find(resolved, a, args, context):
    """
    Process $(find PKG).

    Resolves to the share folder of the package
    :returns: updated resolved argument, ``str``
    :raises: :exc:SubstitutionException: if PKG invalidly specified
    """
    #print(resolved, a, args, context)
#    if len(args) != 1:
#        raise SubstitutionException(
#            '$(find pkg) accepts exactly one argument [%s]' % a)
#    return resolved.replace('$(%s)' % a, _eval_find(args[0]))
    folder = Path(gym_envs.__file__).resolve().parent / env_consts.GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS
    try:
        folder = next(folder.glob("**/"+args[0]))
    except StopIteration:
        raise Exception(f"{args[0]} has not been found anywhere in {folder}")
    return folder


def _eval_arg(name, args):

    try:
        return args[name]
    except KeyError:
        raise ArgException(name)


def _arg(resolved, a, args, context):
    """
    Process $(arg) arg.

    :returns: updated resolved argument, ``str``
    :raises: :exc:`ArgException` If arg invalidly specified
    """
    if len(args) == 0:
        raise SubstitutionException(
            '$(arg var) must specify a variable name [%s]' % (a))
    elif len(args) > 1:
        raise SubstitutionException(
            '$(arg var) may only specify one arg [%s]' % (a))

    if 'arg' not in context:
        context['arg'] = {}
    return resolved.replace('$(%s)' % a, _eval_arg(name=args[0], args=context['arg']))


# Create a dictionary of global symbols that will be available in the eval
# context.  We disable all the builtins, then add back True and False, and also
# add true and false for convenience (because we accept those lower-case strings
# as boolean values in XML).
_eval_dict = {
    'true': True, 'false': False,
    'True': True, 'False': False,
    '__builtins__': {k: __builtins__[k] for k in ['list', 'dict', 'map', 'str', 'float', 'int']},
    'env': _eval_env,
    'optenv': _eval_optenv,
    #'find': _eval_find
}
# also define all math symbols and functions
_eval_dict.update(math.__dict__)


def convert_value(value, type_):
    """
    Convert a value from a string representation into the specified type.

    @param value: string representation of value
    @type  value: str
    @param type_: int, double, string, bool, or auto
    @type  type_: str
    @raise ValueError: if parameters are invalid
    """
    type_ = type_.lower()
    # currently don't support XML-RPC date, dateTime, maps, or list
    # types
    if type_ == 'auto':
        # attempt numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        # bool
        lval = value.lower()
        if lval == 'true' or lval == 'false':
            return convert_value(value, 'bool')
        # string
        return value
    elif type_ == 'str' or type_ == 'string':
        return value
    elif type_ == 'int':
        return int(value)
    elif type_ == 'double':
        return float(value)
    elif type_ == 'bool' or type_ == 'boolean':
        value = value.lower().strip()
        if value == 'true' or value == '1':
            return True
        elif value == 'false' or value == '0':
            return False
        raise ValueError("%s is not a '%s' type" % (value, type_))
    elif type_ == 'yaml':
        try:
            import yaml
            return yaml.load(value)
        except yaml.parser.ParserError as e:
            raise ValueError(e)
    else:
        raise ValueError("Unknown type '%s'" % type_)


class _DictWrapper(object):

    def __init__(self, args, functions):
        self._args = args
        self._functions = functions

    def __getitem__(self, key):
        try:
            return self._functions[key]
        except KeyError:
            return convert_value(self._args[key], 'auto')


def _eval(s, context):

    if 'arg' not in context:
        context['arg'] = {}

    # inject arg context
    def _eval_arg_context(name):
        return convert_value(_eval_arg(name, args=context['arg']), 'auto')

    # inject dirname context
    def _eval_dirname_context():
        return _eval_dirname(context['filename'])

    functions = {
        'arg': _eval_arg_context,
        'dirname': _eval_dirname_context
    }
    functions.update(_eval_dict)

    # ignore values containing double underscores (for safety)
    # http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
    if s.find('__') >= 0:
        raise SubstitutionException(
            '$(eval ...) may not contain double underscore expressions')
    return str(eval(s, {}, _DictWrapper(context['arg'], functions)))


def resolve_args(arg_str, context=None, filename=None):
    """
    Resolve substitution args (see wiki spec U{http://ros.org/wiki/roslaunch}).

    @param arg_str: string to resolve zero or more substitution args in.
        arg_str may be None, in which case resolve_args will return None
    @type  arg_str: str
    @param context dict: (optional) dictionary for storing results of the 'arg' substitution args.
        If no context is provided, a new one will be created for each call. Values for the 'arg'
        context should be stored as a dictionary in the 'arg' key.
    @type  context: dict

    @return str: arg_str with substitution args resolved
    @rtype:  str
    @raise SubstitutionException: if there is an error resolving substitution args
    """
    if context is None:
        context = {}
    if not arg_str:
        return arg_str
    # special handling of $(eval ...)
    if arg_str.startswith('$(eval ') and arg_str.endswith(')'):
        return _eval(arg_str[7:-1], context)
    # first resolve variables like 'env' and 'arg'
    commands = {
        'env': _env,
        'optenv': _optenv,
        'dirname': _dirname,
        'arg': _arg,
        'find': _find,
    }
    resolved = _resolve_args(arg_str, context, commands)
    return resolved


def _resolve_args(arg_str, context, commands):

    valid = ['find', 'env', 'optenv', 'dirname', 'arg']
    resolved = arg_str
    for a in _collect_args(arg_str):
        splits = [s for s in a.split(' ') if s]
        if not splits[0] in valid:
            raise SubstitutionException('Unknown substitution command [%s]. '
                                        'Valid commands are %s' % (a, valid))
        command = splits[0]
        args = splits[1:]
        if command in commands:
            resolved = commands[command](resolved, a, args, context)
    return resolved


_OUT = 0
_DOLLAR = 1
_LP = 2
_IN = 3


def _collect_args(arg_str):
    """
    State-machine parser for resolve_args.

    Substitution args are of the form:
    $(find package_name)/scripts/foo.py $(export some/attribute blar) non-relevant stuff

    @param arg_str: argument string to parse args from
    @type  arg_str: str
    @raise SubstitutionException: if args are invalidly specified
    @return: list of arguments
    @rtype: [str]
    """
    buff = StringIO()
    args = []
    state = _OUT
    for c in arg_str:
        # No escapes supported
        if c == '$':
            if state == _OUT:
                state = _DOLLAR
            elif state == _DOLLAR:
                pass
            else:
                raise SubstitutionException('Dollar signs "$" cannot be '
                                            'inside of substitution args [%s]' % arg_str)
        elif c == '(':
            if state == _DOLLAR:
                state = _LP
            elif state != _OUT:
                raise SubstitutionException('Invalid left parenthesis "(" '
                                            'in substitution args [%s]' % arg_str)
        elif c == ')':
            if state == _IN:
                # save contents of collected buffer
                args.append(buff.getvalue())
                buff.truncate(0)
                buff.seek(0)
                state = _OUT
            else:
                state = _OUT
        elif state == _DOLLAR:
            # left paren must immediately follow dollar sign to enter _IN state
            state = _OUT
        elif state == _LP:
            state = _IN

        if state == _IN:
            buff.write(c)
    return args



# Dictionary of substitution args
substitution_args_context = {}


# Stack of currently processed files
filestack = []


def push_file(filename):
    """
    Push a new filename to the filestack.
    Instead of directly modifying filestack, a deep-copy is created and modified,
    while the old filestack is returned.
    This allows to store the filestack that was active when a macro or property is defined
    """
    global filestack
    oldstack = filestack
    filestack = deepcopy(filestack)
    filestack.append(filename)
    return oldstack


def restore_filestack(oldstack):
    global filestack
    filestack = oldstack


def abs_filename_spec(filename_spec):
    """
    Prepend the dirname of the currently processed file
    if filename_spec is not yet absolute
    """
    if not os.path.isabs(filename_spec):
        parent_filename = filestack[-1]
        basedir = os.path.dirname(parent_filename) if parent_filename else '.'
        return os.path.join(basedir, filename_spec)
    return filename_spec


class YamlListWrapper(list):
    """Wrapper class for yaml lists to allow recursive inheritance of wrapper property"""
    @staticmethod
    def wrap(item):
        """This static method, used by both YamlListWrapper and YamlDictWrapper,
           dispatches to the correct wrapper class depending on the type of yaml item"""
        if isinstance(item, dict):
            return YamlDictWrapper(item)
        elif isinstance(item, list):
            return YamlListWrapper(item)
        else: # scalar
            return item

    def __getitem__(self, idx):
        return YamlListWrapper.wrap(super(YamlListWrapper, self).__getitem__(idx))


class YamlDictWrapper(dict):
    """Wrapper class providing dotted access to dict items"""
    def __getattr__(self, item):
        try:
            return YamlListWrapper.wrap(super(YamlDictWrapper, self).__getitem__(item))
        except KeyError:
            raise XacroException("No such key: '{}'".format(item))

    __getitem__ = __getattr__


def construct_angle_radians(loader, node):
    """utility function to construct radian values from yaml"""
    value = loader.construct_scalar(node).strip()
    try:
        return float(eval(value, global_symbols))
    except SyntaxError as e:
        raise XacroException("invalid expression: %s" % value)


def construct_angle_degrees(loader, node):
    """utility function for converting degrees into radians from yaml"""
    value = loader.construct_scalar(node)
    try:
        return math.radians(float(value))
    except ValueError:
        raise XacroException("invalid degree value: %s" % value)


def load_yaml(filename):
    try:
        import yaml
        yaml.SafeLoader.add_constructor(u'!radians', construct_angle_radians)
        yaml.SafeLoader.add_constructor(u'!degrees', construct_angle_degrees)
    except Exception:
        raise XacroException("yaml support not available; install python-yaml")

    filename = abs_filename_spec(filename)
    f = open(filename)
    oldstack = push_file(filename)
    try:
        return YamlListWrapper.wrap(yaml.safe_load(f))
    finally:
        f.close()
        restore_filestack(oldstack)
        global all_includes
        all_includes.append(filename)


# global symbols dictionary
# taking simple security measures to forbid access to __builtins__
# only the very few symbols explicitly listed are allowed
# for discussion, see: http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
global_symbols = {'__builtins__': {k: __builtins__[k] for k in
                                   ['list', 'dict', 'map', 'len', 'str', 'float', 'int',
                                    'True', 'False', 'min', 'max', 'round']}}
# also define all math symbols and functions
global_symbols.update(math.__dict__)
# expose load_yaml and abs_filename
global_symbols.update(dict(load_yaml=load_yaml, abs_filename=abs_filename_spec))


class XacroException(Exception):
    """
    XacroException allows to wrap another exception (exc) and to augment
    its error message: prefixing with msg and suffixing with suffix.
    str(e) finally prints: msg str(exc) suffix
    """

    def __init__(self, msg=None, suffix=None, exc=None, macro=None):
        super(XacroException, self).__init__(msg)
        self.suffix = suffix
        self.exc = exc
        self.macros = [] if macro is None else [macro]

    def __str__(self):
        items = [super(XacroException, self).__str__(), self.exc, self.suffix]
        return ' '.join([s for s in [str(e) for e in items] if s not in ['', 'None']])


verbosity = 1

def check_attrs(tag, required, optional):
    """
    Helper routine to fetch required and optional attributes
    and complain about any additional attributes.
    :param tag (xml.dom.Element): DOM element node
    :param required [str]: list of required attributes
    :param optional [str]: list of optional attributes
    """
    result = reqd_attrs(tag, required)
    result.extend(opt_attrs(tag, optional))
    allowed = required + optional
    extra = [a for a in tag.attributes.keys() if a not in allowed and not a.startswith("xmlns:")]
    if extra:
        warning("%s: unknown attribute(s): %s" % (tag.nodeName, ', '.join(extra)))
        if verbosity > 0:
            print_location(filestack)
    return result


# deprecate non-namespaced use of xacro tags (issues #41, #59, #60)
def deprecated_tag(tag_name=None, _issued=[False]):
    if _issued[0]:
        return

    if verbosity > 0:
        _issued[0] = True
        warning("Deprecated: xacro tag '{}' w/o 'xacro' xml namespace prefix (will be forbidden in F-turtle)".format(tag_name))
        print_location(filestack)
        message("""Use the following command to fix incorrect tag usage:
find . -iname "*.xacro" | xargs sed -i 's#<\([/]\\?\)\(if\|unless\|include\|arg\|property\|macro\|insert_block\)#<\\1xacro:\\2#g'""")
        print(file=sys.stderr)


# require xacro namespace?
allow_non_prefixed_tags = True


def check_deprecated_tag(tag_name):
    """
    Check whether tagName starts with xacro prefix. If not, issue a warning.
    :param tag_name:
    :return: True if tagName is accepted as xacro tag
             False if tagName doesn't start with xacro prefix, but the prefix is required
    """
    if tag_name.startswith('xacro:'):
        return True
    else:
        if allow_non_prefixed_tags:
            deprecated_tag(tag_name)
        return allow_non_prefixed_tags


class Macro(object):
    def __init__(self):
        self.body = None  # original xml.dom.Node
        self.params = []  # parsed parameter names
        self.defaultmap = {}  # default parameter values
        self.history = []  # definition history


def eval_extension(s):
    if s == '$(cwd)':
        return os.getcwd()
    try:
        #from substitution_args import resolve_args, ArgException, PackageNotFoundError
        return resolve_args(s, context=substitution_args_context)
    except ImportError as e:
        raise XacroException("substitution args not supported: ", exc=e)
    #except ArgException as e:
        #raise XacroException("Undefined substitution argument", exc=e)
    #except PackageNotFoundError as e:
        #raise XacroException('package not found:', exc=e)


class Table(dict):
    def __init__(self, parent=None):
        dict.__init__(self)
        self.parent = parent
        self.unevaluated = set()  # set of unevaluated variables
        self.recursive = []  # list of currently resolved vars (to resolve recursive definitions)
        # the following variables are for debugging / checking only
        self.depth = self.parent.depth + 1 if self.parent else 0

    @staticmethod
    def _eval_literal(value):
        if isinstance(value, str):
            # remove single quotes from escaped string
            if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
                return value[1:-1]
            # Try to evaluate as number literal or boolean.
            # This is needed to handle numbers in property definitions as numbers, not strings.
            # python3 ignores/drops underscores in number literals (due to PEP515).
            # Here, we want to handle literals with underscores as plain strings.
            if '_' in value:
                return value
            for f in [int, float, lambda x: get_boolean_value(x, None)]:  # order of types is important!
                try:
                    return f(value)
                except Exception:
                    pass
        return value

    def _resolve_(self, key):
        # lazy evaluation
        if key in self.unevaluated:
            if key in self.recursive:
                raise XacroException("recursive variable definition: %s" %
                                     " -> ".join(self.recursive + [key]))
            self.recursive.append(key)
            dict.__setitem__(self, key, self._eval_literal(eval_text(dict.__getitem__(self, key), self)))
            self.unevaluated.remove(key)
            self.recursive.remove(key)

        # return evaluated result
        value = dict.__getitem__(self, key)
        if (verbosity > 2 and self.parent is None) or verbosity > 3:
            print("{indent}use {key}: {value} ({loc})".format(
                indent=self.depth * ' ', key=key, value=value, loc=filestack[-1]), file=sys.stderr)
        return value

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return self._resolve_(key)
        elif self.parent:
            return self.parent[key]
        else:
            return global_symbols[key]

    def _setitem(self, key, value, unevaluated):
        if key in global_symbols:
            warning("redefining global property: %s" % key)
            print_location(filestack)

        value = self._eval_literal(value)
        dict.__setitem__(self, key, value)
        if unevaluated and isinstance(value, str):
            # literal evaluation failed: re-evaluate lazily at first access
            self.unevaluated.add(key)
        elif key in self.unevaluated:
            # all other types cannot be evaluated
            self.unevaluated.remove(key)
        if (verbosity > 2 and self.parent is None) or verbosity > 3:
            print("{indent}set {key}: {value} ({loc})".format(
                indent=self.depth * ' ', key=key, value=value, loc=filestack[-1]), file=sys.stderr)

    def __setitem__(self, key, value):
        self._setitem(key, value, unevaluated=True)

    def __contains__(self, key):
        return \
            dict.__contains__(self, key) or \
            (self.parent and key in self.parent)

    def __str__(self):
        s = dict.__str__(self)
        if self.parent is not None:
            s += "\n  parent: "
            s += str(self.parent)
        return s

    def root(self):
        p = self
        while p.parent is not None:
            p = p.parent
        return p


class NameSpace(object):
    # dot access (namespace.property) is forwarded to getitem()
    def __getattr__(self, item):
        return self.__getitem__(item)


class PropertyNameSpace(Table, NameSpace):
    def __init__(self, parent=None):
        super(PropertyNameSpace, self).__init__(parent)


class MacroNameSpace(dict, NameSpace):
    def __init__(self, *args, **kwargs):
        super(MacroNameSpace, self).__init__(*args, **kwargs)


class QuickLexer(object):
    def __init__(self, *args, **kwargs):
        if args:
            # copy attributes + variables from other instance
            other = args[0]
            self.__dict__.update(other.__dict__)
        else:
            self.res = []
            for k, v in kwargs.items():
                self.__setattr__(k, len(self.res))
                self.res.append(re.compile(v))
        self.str = ""
        self.top = None

    def lex(self, str):
        self.str = str
        self.top = None
        self.next()

    def peek(self):
        return self.top

    def next(self):
        result = self.top
        self.top = None
        if not self.str:  # empty string
            return result
        for i in range(len(self.res)):
            m = self.res[i].match(self.str)
            if m:
                self.top = (i, m.group(0))
                self.str = self.str[m.end():]
                return result
        raise XacroException('invalid expression: ' + self.str)


all_includes = []
include_no_matches_msg = """Include tag's filename spec \"{}\" matched no files."""


def is_include(elt):
    # Xacro should not use plain 'include' tags but only namespaced ones. Causes conflicts with
    # other XML elements including Gazebo's <gazebo> extensions
    if elt.tagName not in ['xacro:include', 'include']:
        return False

    # Temporary fix for ROS Hydro and the xacro include scope problem
    if elt.tagName == 'include':
        # check if there is any element within the <include> tag. mostly we are concerned
        # with Gazebo's <uri> element, but it could be anything. also, make sure the child
        # nodes aren't just a single Text node, which is still considered a deprecated
        # instance
        if elt.childNodes and not (len(elt.childNodes) == 1 and
                                   elt.childNodes[0].nodeType == elt.TEXT_NODE):
            # this is not intended to be a xacro element, so we can ignore it
            return False
        else:
            # throw a deprecated warning
            return check_deprecated_tag(elt.tagName)
    return True


def get_include_files(filename_spec, symbols):
    try:
        filename_spec = abs_filename_spec(eval_text(filename_spec, symbols))
    except XacroException as e:
        if e.exc and isinstance(e.exc, NameError) and symbols is None:
            raise XacroException('variable filename is supported with in-order option only')
        else:
            raise

    if re.search('[*[?]+', filename_spec):
        # Globbing behaviour
        filenames = sorted(glob.glob(filename_spec))
        if len(filenames) == 0:
            warning(include_no_matches_msg.format(filename_spec))
    else:
        # Default behaviour
        filenames = [filename_spec]

    for filename in filenames:
        global all_includes
        all_includes.append(filename)
        yield filename


def import_xml_namespaces(parent, attributes):
    """import all namespace declarations into parent"""
    for name, value in attributes.items():
        if name.startswith('xmlns:'):
            oldAttr = parent.getAttributeNode(name)
            if oldAttr and oldAttr.value != value:
                warning("inconsistent namespace redefinitions for {name}:"
                        "\n old: {old}\n new: {new} ({new_file})".format(
                            name=name, old=oldAttr.value, new=value,
                            new_file=filestack[-1]))
            else:
                parent.setAttribute(name, value)


def process_include(elt, macros, symbols, func):
    included = []
    filename_spec, namespace_spec, optional = check_attrs(elt, ['filename'], ['ns', 'optional'])
    
    if namespace_spec:
        try:
            namespace_spec = eval_text(namespace_spec, symbols)
            macros[namespace_spec] = ns_macros = MacroNameSpace()
            symbols[namespace_spec] = ns_symbols = PropertyNameSpace()
        except TypeError:
            raise XacroException('namespaces are supported with in-order option only')
    else:
        ns_macros = macros
        ns_symbols = symbols

    optional = get_boolean_value(optional, None)

    if first_child_element(elt):
        warning("Child elements of a <xacro:include> tag are ignored")
        if verbosity > 0:
            print_location(filestack)
    for filename in get_include_files(filename_spec, symbols):
        try:
            # extend filestack
            oldstack = push_file(filename)
            include = parse(None, filename).documentElement

            # recursive call to func
            func(include, ns_macros, ns_symbols)
            included.append(include)
            import_xml_namespaces(elt.parentNode, include.attributes)
            # restore filestack
            restore_filestack(oldstack)
        except XacroException as e:
            if e.exc and isinstance(e.exc, IOError) and optional is True:
                continue
            else:
                raise

    remove_previous_comments(elt)
    # replace the include tag with the nodes of the included file(s)
    replace_node(elt, by=included, content_only=True)


def is_valid_name(name):
    """
    Checks whether name is a valid property or macro identifier.
    With python-based evaluation, we need to avoid name clashes with python keywords.
    """
    # Resulting AST of simple identifier is <Module [<Expr <Name "foo">>]>
    try:
        root = ast.parse(name)

        if isinstance(root, ast.Module) and \
           len(root.body) == 1 and isinstance(root.body[0], ast.Expr) and \
           isinstance(root.body[0].value, ast.Name) and root.body[0].value.id == name:
            return True
    except SyntaxError:
        pass

    return False


re_macro_arg = re.compile(r'''\s*([^\s:=]+?):?=(\^\|?)?((?:(?:'[^']*')?[^\s'"]*?)*)(?:\s+|$)(.*)''')
#                           space   param    :=   ^|   <--      default      -->   space    rest


def parse_macro_arg(s):
    """
    parse the first param spec from a macro parameter string s
    accepting the following syntax: <param>[:=|=][^|]<default>
    :param s: param spec string
    :return: param, (forward, default), rest-of-string
             forward will be either param or None (depending on whether ^ was specified)
             default will be the default string or None
             If there is no default spec at all, the middle pair will be replaced by None
    """
    m = re_macro_arg.match(s)
    if m:
        # there is a default value specified for param
        param, forward, default, rest = m.groups()
        if not default:
            default = None
        return param, (param if forward else None, default), rest
    else:
        # there is no default specified at all
        result = s.lstrip().split(None, 1)
        return result[0], None, result[1] if len(result) > 1 else ''


def grab_macro(elt, macros):
    assert(elt.tagName in ['macro', 'xacro:macro'])
    remove_previous_comments(elt)

    name, params = check_attrs(elt, ['name'], ['params'])
    if name == 'call':
        raise XacroException("Invalid use of macro name 'call'")
    if name.find('.') != -1:
        raise XacroException("macro names must not contain '.' (reserved for namespaces): %s" % name)
    if name.startswith('xacro:'):
        warning("macro names must not contain prefix 'xacro:': %s" % name)
        name = name[6:]  # drop 'xacro:' prefix

    # fetch existing or create new macro definition
    macro = macros.get(name, Macro())
    # append current filestack to history
    macro.history.append(filestack)
    macro.body = elt

    # parse params and their defaults
    macro.params = []
    macro.defaultmap = {}
    while params:
        param, value, params = parse_macro_arg(params)
        macro.params.append(param)
        if value is not None:
            macro.defaultmap[param] = value  # parameter with default

    macros[name] = macro
    replace_node(elt, by=None)


def grab_property(elt, table):
    assert(elt.tagName in ['property', 'xacro:property'])
    remove_previous_comments(elt)

    name, value, default, scope = check_attrs(elt, ['name'], ['value', 'default', 'scope'])
    if not is_valid_name(name):
        raise XacroException('Property names must be valid python identifiers: ' + name)
    if value is not None and default is not None:
        raise XacroException('Property cannot define both a default and a value: ' + name)

    if default is not None:
        if scope is not None:
            warning("%s: default property value can only be defined on local scope" % name)
        if name not in table:
            value = default
        else:
            replace_node(elt, by=None)
            return

    if value is None:
        name = '**' + name
        value = elt  # debug

    replace_node(elt, by=None)

    if scope and scope == 'global':
        target_table = table.root()
        unevaluated = False
    elif scope and scope == 'parent':
        if table.parent is not None:
            target_table = table.parent
            unevaluated = False
        else:
            warning("%s: no parent scope at global scope " % name)
            return  # cannot store the value, no reason to evaluate it
    else:
        target_table = table
        unevaluated = True

    if not unevaluated and isinstance(value, str):
        value = eval_text(value, table)

    target_table._setitem(name, value, unevaluated=unevaluated)


LEXER = QuickLexer(DOLLAR_DOLLAR_BRACE=r"^\$\$+(\{|\()",  # multiple $ in a row, followed by { or (
                   EXPR=r"^\$\{[^\}]*\}",        # stuff starting with ${
                   EXTENSION=r"^\$\([^\)]*\)",   # stuff starting with $(
                   TEXT=r"[^$]+|\$[^{($]+|\$$")  # any text w/o $  or  $ following any chars except {($  or  single $


# evaluate text and return typed value
def eval_text(text, symbols):
    def handle_expr(s):
        try:
            return eval(eval_text(s, symbols), symbols)
        except Exception as e:
            # re-raise as XacroException to add more context
            raise XacroException(exc=e,
                                 suffix=os.linesep + "when evaluating expression '%s'" % s)

    def handle_extension(s):
        return eval_extension("$(%s)" % eval_text(s, symbols))

    results = []
    lex = QuickLexer(LEXER)
    lex.lex(text)
    while lex.peek():
        id = lex.peek()[0]
        if id == lex.EXPR:
            results.append(handle_expr(lex.next()[1][2:-1]))
        elif id == lex.EXTENSION:
            results.append(handle_extension(lex.next()[1][2:-1]))
        elif id == lex.TEXT:
            results.append(lex.next()[1])
        elif id == lex.DOLLAR_DOLLAR_BRACE:
            results.append(lex.next()[1][1:])
    # return single element as is, i.e. typed
    if len(results) == 1:
        return results[0]
    # otherwise join elements to a string
    else:
        return ''.join(map(str, results))


def eval_default_arg(forward_variable, default, symbols, macro):
    if forward_variable is None:
        return eval_text(default, symbols)
    try:
        return symbols[forward_variable]
    except KeyError:
        if default is not None:
            return eval_text(default, symbols)
        else:
            raise XacroException("Undefined property to forward: " + forward_variable, macro=macro)


def handle_dynamic_macro_call(node, macros, symbols):
    name, = reqd_attrs(node, ['macro'])
    if not name:
        raise XacroException("xacro:call is missing the 'macro' attribute")
    name = str(eval_text(name, symbols))

    # remove 'macro' attribute and rename tag with resolved macro name
    node.removeAttribute('macro')
    node.tagName = 'xacro:' + name
    # forward to handle_macro_call
    try:
        return handle_macro_call(node, macros, symbols)
    except KeyError:
        raise XacroException("unknown macro name '%s' in xacro:call" % name)


def resolve_macro(fullname, macros):
    # split name into namespaces and real name
    namespaces = fullname.split('.')
    name = namespaces.pop(-1)

    def _resolve(namespaces, name, macros):
        # traverse namespaces to actual macros dict
        for ns in namespaces:
            macros = macros[ns]
        return macros[name]

    # try fullname and (namespaces, name) in this order
    try:
        return _resolve([], fullname, macros)
    except KeyError:
        if namespaces:
            return _resolve(namespaces, name, macros)
        else:
            raise


def handle_macro_call(node, macros, symbols):
    if node.tagName.startswith('xacro:'):
        name = node.tagName[6:]  # strip off 'xacro:' prefix
    elif allow_non_prefixed_tags:
        name = node.tagName
    else:  # require prefixed macro names
        return False

    try:
        m = resolve_macro(name, macros)
        if name is node.tagName:  # no xacro prefix provided?
            deprecated_tag(name)
        body = m.body.cloneNode(deep=True)

    except KeyError:
        # TODO If deprecation runs out, this test should be moved up front
        if node.tagName == 'xacro:call':
            return handle_dynamic_macro_call(node, macros, symbols)
        return False  # no macro

    # Expand the macro
    scoped = Table(symbols)  # new local name space for macro evaluation
    params = m.params[:]  # deep copy macro's params list
    for name, value in node.attributes.items():
        if name not in params:
            raise XacroException("Invalid parameter '%s'" % str(name), macro=m)
        params.remove(name)
        scoped._setitem(name, eval_text(value, symbols), unevaluated=False)
        node.setAttribute(name, "")  # suppress second evaluation in eval_all()

    # Evaluate block parameters in node
    eval_all(node, macros, symbols)

    # Fetch block parameters, in order
    block = first_child_element(node)
    for param in params[:]:
        if param[0] == '*':
            if not block:
                raise XacroException("Not enough blocks", macro=m)
            params.remove(param)
            scoped[param] = block
            block = next_sibling_element(block)

    if block is not None:
        raise XacroException("Unused block '%s'" % block.tagName, macro=m)

    # Try to load defaults for any remaining non-block parameters
    for param in params[:]:
        # block parameters are not supported for defaults
        if param[0] == '*':
            continue

        # get default
        name, default = m.defaultmap.get(param, (None, None))
        if name is not None or default is not None:
            scoped._setitem(param, eval_default_arg(name, default, symbols, m), unevaluated=False)
            params.remove(param)

    if params:
        raise XacroException("Undefined parameters [%s]" % ",".join(params), macro=m)

    try:
        eval_all(body, macros, scoped)
    except Exception as e:
        # fill in macro call history for nice error reporting
        if hasattr(e, 'macros'):
            e.macros.append(m)
        else:
            e.macros = [m]
        raise

    # Replaces the macro node with the expansion
    remove_previous_comments(node)
    replace_node(node, by=body, content_only=True)
    return True


def get_boolean_value(value, condition):
    """
    Return a boolean value that corresponds to the given Xacro condition value.
    Values "true", "1" and "1.0" are supposed to be True.
    Values "false", "0" and "0.0" are supposed to be False.
    All other values raise an exception.

    :param value: The value to be evaluated. The value has to already be evaluated by Xacro.
    :param condition: The original condition text in the XML.
    :return: The corresponding boolean value, or a Python expression that, converted to boolean, corresponds to it.
    :raises ValueError: If the condition value is incorrect.
    """
    try:
        if isinstance(value, str):
            if value == 'true' or value == 'True':
                return True
            elif value == 'false' or value == 'False':
                return False
            else:
                return bool(int(value))
        else:
            return bool(value)
    except Exception:
        raise XacroException("Xacro conditional '%s' evaluated to '%s', "
                             "which is not a boolean expression." % (condition, value))


_empty_text_node = xml.dom.minidom.getDOMImplementation().createDocument(None, "dummy", None).createTextNode('\n\n')


def remove_previous_comments(node):
    """remove consecutive comments in front of the xacro-specific node"""
    next = node.nextSibling
    previous = node.previousSibling
    while previous:
        if previous.nodeType == xml.dom.Node.TEXT_NODE and \
                previous.data.isspace() and previous.data.count('\n') <= 1:
            previous = previous.previousSibling  # skip a single empty text node (max 1 newline)

        if previous and previous.nodeType == xml.dom.Node.COMMENT_NODE:
            comment = previous
            previous = previous.previousSibling
            node.parentNode.removeChild(comment)
        else:
            # insert empty text node to stop removing of comments in future calls
            # actually this moves the singleton instance to the new location
            if next and _empty_text_node != next:
                node.parentNode.insertBefore(_empty_text_node, next)
            return


def eval_all(node, macros, symbols):
    """Recursively evaluate node, expanding macros, replacing properties, and evaluating expressions"""
    # evaluate the attributes
    for name, value in node.attributes.items():
        if name.startswith('xacro:'):  # remove xacro:* attributes
            node.removeAttribute(name)
        else:
            result = str(eval_text(value, symbols))
            node.setAttribute(name, result)

    # remove xacro namespace definition
    try:
        node.removeAttribute('xmlns:xacro')
    except xml.dom.NotFoundErr:
        pass

    node = node.firstChild
    while node:
        next = node.nextSibling
        if node.nodeType == xml.dom.Node.ELEMENT_NODE:
            if node.tagName in ['insert_block', 'xacro:insert_block'] \
                    and check_deprecated_tag(node.tagName):
                name, = check_attrs(node, ['name'], [])

                if ("**" + name) in symbols:
                    # Multi-block
                    block = symbols['**' + name]
                    content_only = True
                elif ("*" + name) in symbols:
                    # Single block
                    block = symbols['*' + name]
                    content_only = False
                else:
                    raise XacroException("Undefined block '%s'" % name)

                # cloning block allows to insert the same block multiple times
                block = block.cloneNode(deep=True)
                # recursively evaluate block
                eval_all(block, macros, symbols)
                replace_node(node, by=block, content_only=content_only)

            elif is_include(node):
                process_include(node, macros, symbols, eval_all)

            elif node.tagName in ['property', 'xacro:property'] \
                    and check_deprecated_tag(node.tagName):
                grab_property(node, symbols)

            elif node.tagName in ['macro', 'xacro:macro'] \
                    and check_deprecated_tag(node.tagName):
                grab_macro(node, macros)

            elif node.tagName in ['arg', 'xacro:arg'] \
                    and check_deprecated_tag(node.tagName):
                name, default = check_attrs(node, ['name', 'default'], [])
                if name not in substitution_args_context['arg']:
                    substitution_args_context['arg'][name] = eval_text(default, symbols)

                remove_previous_comments(node)
                replace_node(node, by=None)

            elif node.tagName == 'xacro:element':
                name = eval_text(*reqd_attrs(node, ['xacro:name']), symbols=symbols)
                if not name:
                    raise XacroException("xacro:element: empty name")

                node.removeAttribute('xacro:name')
                node.nodeName = node.tagName = name
                continue  # re-process the node with new tagName

            elif node.tagName == 'xacro:attribute':
                name, value = [eval_text(a, symbols) for a in reqd_attrs(node, ['name', 'value'])]
                if not name:
                    raise XacroException("xacro:attribute: empty name")

                node.parentNode.setAttribute(name, value)
                replace_node(node, by=None)

            elif node.tagName in ['if', 'xacro:if', 'unless', 'xacro:unless'] \
                    and check_deprecated_tag(node.tagName):
                remove_previous_comments(node)
                cond, = check_attrs(node, ['value'], [])
                keep = get_boolean_value(eval_text(cond, symbols), cond)
                if node.tagName in ['unless', 'xacro:unless']:
                    keep = not keep

                if keep:
                    eval_all(node, macros, symbols)
                    replace_node(node, by=node, content_only=True)
                else:
                    replace_node(node, by=None)

            elif handle_macro_call(node, macros, symbols):
                pass  # handle_macro_call does all the work of expanding the macro

            else:
                # these are the non-xacro tags
                if node.tagName.startswith("xacro:"):
                    raise XacroException("unknown macro name: %s" % node.tagName[6:])

                eval_all(node, macros, symbols)

        # TODO: Also evaluate content of COMMENT_NODEs?
        elif node.nodeType == xml.dom.Node.TEXT_NODE:
            node.data = str(eval_text(node.data, symbols))

        node = next


def parse(inp, filename=None):
    """
    Parse input or filename into a DOM tree.
    If inp is None, open filename and load from there.
    Otherwise, parse inp, either as string or file object.
    If inp is already a DOM tree, this function is a noop.
    :return:xml.dom.minidom.Document
    :raise: xml.parsers.expat.ExpatError
    """
    f = None
    if inp is None:
        try:
            inp = f = open(filename)
        except IOError as e:
            # do not report currently processed file as "in file ..."
            filestack.pop()
            raise XacroException(e.strerror + ": " + e.filename, exc=e)

    try:
        if isinstance(inp, str):
            return xml.dom.minidom.parseString(inp)
        elif hasattr(inp, 'read'):
            return xml.dom.minidom.parse(inp)
        return inp

    finally:
        if f:
            f.close()


def process_doc(doc, mappings=None, xacro_ns=True, **kwargs):
    global verbosity
    verbosity = kwargs.get('verbosity', verbosity)

    # set substitution args
    substitution_args_context['arg'] = {} if mappings is None else mappings

    global allow_non_prefixed_tags
    allow_non_prefixed_tags = xacro_ns = xacro_ns

    # if not yet defined: initialize filestack
    if not filestack:
        restore_filestack([None])

    macros = {}
    symbols = Table()

    # apply xacro:targetNamespace as global xmlns (if defined)
    targetNS = doc.documentElement.getAttribute('xacro:targetNamespace')
    if targetNS:
        doc.documentElement.removeAttribute('xacro:targetNamespace')
        doc.documentElement.setAttribute('xmlns', targetNS)

    eval_all(doc.documentElement, macros, symbols)

    # reset substitution args
    substitution_args_context['arg'] = {}


def open_output(output_filename):
    if output_filename is None:
        return sys.stdout
    else:
        dir_name = os.path.dirname(output_filename)
        if dir_name:
            try:
                os.makedirs(dir_name)
            except os.error:
                # errors occur when dir_name exists or creation failed
                # ignore error here; opening of file will fail if directory is still missing
                pass

        try:
            return open(output_filename, 'w')
        except IOError as e:
            raise XacroException("Failed to open output:", exc=e)


def print_location(filestack, err=None, file=sys.stderr):
    macros = getattr(err, 'macros', []) if err else []
    msg = 'when instantiating macro:'
    for m in macros:
        name = m.body.getAttribute('name')
        location = '(%s)' % m.history[-1][-1]
        print(msg, name, location, file=file)
        msg = 'instantiated from:'

    msg = 'in file:' if macros else 'when processing file:'
    for f in reversed(filestack):
        if f is None:
            f = 'string'
        print(msg, f, file=file)
        msg = 'included from:'


def process_file(input_file_name, **kwargs):
    """main processing pipeline"""
    # initialize file stack for error-reporting
    restore_filestack([input_file_name])
    # parse the document into a xml.dom tree
    doc = parse(None, input_file_name)
    # perform macro replacement
    process_doc(doc, **kwargs)

    # add xacro auto-generated banner
    banner = [xml.dom.minidom.Comment(c) for c in
              [" %s " % ('=' * 83),
               " |    This document was autogenerated by xacro from %-30s | " % input_file_name,
               " |    EDITING THIS FILE BY HAND IS NOT RECOMMENDED  %-30s | " % "",
               " %s " % ('=' * 83)]]
    first = doc.firstChild
    for comment in banner:
        doc.insertBefore(comment, first)

    return doc


def _process(input_file_name, opts):
    try:
        # open and process file
        doc = process_file(input_file_name, **opts)
        # open the output file
        out = open_output(opts['output'])


    
    except Exception as e:
        msg = str(e)
        if not msg:
            msg = repr(e)
        error(msg)
        if verbosity > 0:
            print_location(filestack, e)
        if verbosity > 1:
            print(file=sys.stderr)  # add empty separator line before error
            raise  # create stack trace
        else:
            sys.exit(2)  # gracefully exit with error condition
    

    if opts['just_deps']:  # only output list of dependencies
        out.write(' '.join(set(all_includes)))
    else:  # write XML output
        out.write(doc.toprettyxml(indent='  '))

    # only close output file, but not stdout
    if opts['output']:
        out.close()


def process(input_file_name, just_deps=False, xacro_ns=True, verbosity=1, mappings={}):
    """Function to be used from python code, returning the processed XML"""
    
    old, sys.stdout = sys.stdout, StringIO()  # temporarily replace sys.stdout with StringIO()
    _process(input_file_name, dict(output=None, just_deps=just_deps, xacro_ns=xacro_ns, verbosity=verbosity, mappings=mappings))
    sys.stdout.seek(0)
    result = sys.stdout.read()
    sys.stdout = old  # restore sys.stdout
    return result
