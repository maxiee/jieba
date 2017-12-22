from __future__ import absolute_import, unicode_literals
__version__ = '0.39'
__license__ = 'MIT'

import re
import os
import sys
import time
import logging
import marshal
import tempfile
import threading
from math import log
from hashlib import md5
from ._compat import *
from . import finalseg

if os.name == 'nt':
    from shutil import move as _replace_file
else:
    _replace_file = os.rename

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)

DICT_WRITING = {}

pool = None

re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$', re.U)

re_eng = re.compile('[a-zA-Z0-9]', re.U)

# \u4E00-\u9FD5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
# \r\n|\s : whitespace characters. Will not be handled.
re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)", re.U)
re_skip_default = re.compile("(\r\n|\s)", re.U)
re_han_cut_all = re.compile("([\u4E00-\u9FD5]+)", re.U)
re_skip_cut_all = re.compile("[^a-zA-Z0-9+#\n]", re.U)

def setLogLevel(log_level):
    global logger
    default_logger.setLevel(log_level)

class Tokenizer(object):

    def __init__(self, dictionary=DEFAULT_DICT):
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            self.dictionary = _get_abs_path(dictionary)
        self.FREQ = {}
        self.total = 0
        self.user_word_tag_tab = {}
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None

    def __repr__(self):
        return '<Tokenizer dictionary=%r>' % self.dictionary

    def gen_pfdict(self, f):
        lfreq = {}
        ltotal = 0
        f_name = resolve_filename(f)
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode('utf-8')
                word, freq = line.split(' ')[:2]
                freq = int(freq)
                lfreq[word] = freq
                ltotal += freq
                for ch in xrange(len(word)):
                    wfrag = word[:ch + 1]
                    if wfrag not in lfreq:
                        lfreq[wfrag] = 0
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        f.close()
        return lfreq, ltotal

    def initialize(self, dictionary=None):
        if dictionary:
            abs_path = _get_abs_path(dictionary)
            if self.dictionary == abs_path and self.initialized:
                return
            else:
                self.dictionary = abs_path
                self.initialized = False
        else:
            abs_path = self.dictionary

        with self.lock:
            try:
                with DICT_WRITING[abs_path]:
                    pass
            except KeyError:
                pass
            if self.initialized:
                return

            default_logger.debug("Building prefix dict from %s ..." % (abs_path or 'the default dictionary'))
            t1 = time.time()
            if self.cache_file:
                cache_file = self.cache_file
            # default dictionary
            elif abs_path == DEFAULT_DICT:
                cache_file = "jieba.cache"
            # custom dictionary
            else:
                cache_file = "jieba.u%s.cache" % md5(
                    abs_path.encode('utf-8', 'replace')).hexdigest()
            cache_file = os.path.join(
                self.tmp_dir or tempfile.gettempdir(), cache_file)
            # prevent absolute path in self.cache_file
            tmpdir = os.path.dirname(cache_file)

            load_from_cache_fail = True
            if os.path.isfile(cache_file) and (abs_path == DEFAULT_DICT or
                os.path.getmtime(cache_file) > os.path.getmtime(abs_path)):
                default_logger.debug(
                    "Loading model from cache %s" % cache_file)
                try:
                    with open(cache_file, 'rb') as cf:
                        self.FREQ, self.total = marshal.load(cf)
                    load_from_cache_fail = False
                except Exception:
                    load_from_cache_fail = True

            if load_from_cache_fail:
                wlock = DICT_WRITING.get(abs_path, threading.RLock())
                DICT_WRITING[abs_path] = wlock
                with wlock:
                    self.FREQ, self.total = self.gen_pfdict(self.get_dict_file())
                    default_logger.debug(
                        "Dumping model to file cache %s" % cache_file)
                    try:
                        # prevent moving across different filesystems
                        fd, fpath = tempfile.mkstemp(dir=tmpdir)
                        with os.fdopen(fd, 'wb') as temp_cache_file:
                            marshal.dump(
                                (self.FREQ, self.total), temp_cache_file)
                        _replace_file(fpath, cache_file)
                    except Exception:
                        default_logger.exception("Dump cache file failed.")

                try:
                    del DICT_WRITING[abs_path]
                except KeyError:
                    pass

            # import pprint
            # pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(self.FREQ)

            self.initialized = True
            default_logger.debug(
                "Loading model cost %.3f seconds." % (time.time() - t1))
            default_logger.debug("Prefix dict has been built succesfully.")

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def calc(self, sentence, DAG, route):
        N = len(sentence)
        # route 里面的两个成员分别代表什么呢?

        route[N] = (0, 0)
        logtotal = log(self.total)
        for idx in xrange(N - 1, -1, -1):
            # 通过取对数, 这里面的乘除操作实际上变成了加减操作
            # 首先最后面这个 DAG[idx] 返回的 DAG Value 可能是个 List
            # for x in DAG[idx] 所 for 的就是这个 List, 这个 List 中至少有一个值, 就是 idx, 表示这个字自己
            # 这样, 当只有一个值时, x 实际跟 idx 是相等的, 此时 sentence[idx:x] (sentence[idx:idx]) 是取不出东西来的. 必须要后面加一才能取出字来
            # 取出来之后, 进行词典查频, 有可能查不到频率,这时  or 1 就是查不到的时候就是 1, 就令这个词频率强制为 1
            # 之后对词频进行 log, 再减去 logtotal, 对数减, 实际上相当于做了触发, 词频 / 总词频, 得到了频率
            # 再乘以什么呢?
            # route[x+1][0] 表示 词路径[x+1,N-1]的最大概率对数,
            # x 是 value list 里面的值
            # max 里面的参数时一整个 tuple (概率, x)
            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or 1) - logtotal + route[x + 1][0], x) for x in DAG[idx])

    def calc2(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(self.total)  # 总词频
        # xrange 与 range 功能完全相同, 所不同的是返回不是一个词组, 而是一个生成器
        # 起始值是 N - 1, 结束值是 -1, 递减值是 -1
        # 实际返回值是 N - 1, N - 2, ... , 0.
        for idx in xrange(N - 1, -1, -1):
            for x in DAG[idx]:  #遍历 DAG
                sentencs = sentence[idx: x + 1]
                freq = self.FREQ.get(sentence[idx: x + 1] or 1)
            pass

    def get_DAG(self, sentence):
        self.check_initialized()
        DAG = {}
        N = len(sentence)
        for k in xrange(N):  # index
            # print(k)
            tmplist = []
            i = k
            frag = sentence[k]
            # print(frag)
            # 找到當前 index 在字典中的最大長度的字的集合
            while i < N and frag in self.FREQ:
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            # 如果字典中不存在
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        import pprint
        pp = pprint.PrettyPrinter()
        pp.pprint(DAG)
        return DAG

    def __cut_all(self, sentence):
        dag = self.get_DAG(sentence)
        old_j = -1
        for k, L in iteritems(dag):
            if len(L) == 1 and k > old_j:
                yield sentence[k:L[0] + 1]
                old_j = L[0]
            else:
                for j in L:
                    if j > k:
                        yield sentence[k:j + 1]
                        old_j = j

    def __cut_DAG_NO_HMM(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                x = y
        if buf:
            yield buf
            buf = ''

    def __cut_DAG(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield buf
                        buf = ''
                    else:
                        if not self.FREQ.get(buf):
                            recognized = finalseg.cut(buf)
                            for t in recognized:
                                yield t
                        else:
                            for elem in buf:
                                yield elem
                        buf = ''
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):
                recognized = finalseg.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def cut(self, sentence, cut_all=False, HMM=True):
        '''
        The main function that segments an entire sentence that contains
        Chinese characters into seperated words.

        Parameter:
            - sentence: The str(unicode) to be segmented.
            - cut_all: Model type. True for full pattern, False for accurate pattern.
            - HMM: Whether to use the Hidden Markov Model.
        '''
        sentence = strdecode(sentence)

        if cut_all:
            re_han = re_han_cut_all
            re_skip = re_skip_cut_all
        else:
            re_han = re_han_default
            re_skip = re_skip_default
        if cut_all:
            cut_block = self.__cut_all
        elif HMM:
            cut_block = self.__cut_DAG
        else:
            cut_block = self.__cut_DAG_NO_HMM
        # 匹配得到中文字符
        # "我来到北京清华大学, 213123, 我来到北京清华大学"
        # ['', '我来到北京清华大学', ', ', '213123', ', ', '我来到北京清华大学', '']
        blocks = re_han.split(sentence)
        print(blocks)
        for blk in blocks:
            if not blk:
                continue
            print(re_han.match(blk))
            if re_han.match(blk): # 滿足正則的走進這裏
                for word in cut_block(blk):  # 最關鍵的算法在 cut_block 中
                    yield word
            else:  # 不滿足正則的走進這裏，例如 ','
                print('run here ' + str(blk))
                # 換行符切分
                tmp = re_skip.split(blk)
                for x in tmp:
                    if re_skip.match(x):  # 如果是換行符，要原樣返回
                        yield x
                    elif not cut_all:  # 否則不是換行符，也不是全模式，把 x 裏面一個一個返回
                        for xx in x:
                            yield xx
                    else:  # 如果是全模式，把 x 整體返回
                        yield x

    def cut_for_search(self, sentence, HMM=True):
        """
        Finer segmentation for search engines.
        """
        words = self.cut(sentence, HMM=HMM)
        for w in words:
            if len(w) > 2:
                for i in xrange(len(w) - 1):
                    gram2 = w[i:i + 2]
                    if self.FREQ.get(gram2):
                        yield gram2
            if len(w) > 3:
                for i in xrange(len(w) - 2):
                    gram3 = w[i:i + 3]
                    if self.FREQ.get(gram3):
                        yield gram3
            yield w

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

    def lcut_for_search(self, *args, **kwargs):
        return list(self.cut_for_search(*args, **kwargs))

    _lcut = lcut
    _lcut_for_search = lcut_for_search

    def _lcut_no_hmm(self, sentence):
        return self.lcut(sentence, False, False)

    def _lcut_all(self, sentence):
        return self.lcut(sentence, True)

    def _lcut_for_search_no_hmm(self, sentence):
        return self.lcut_for_search(sentence, False)

    def get_dict_file(self):
        if self.dictionary == DEFAULT_DICT:
            return get_module_res(DEFAULT_DICT_NAME)
        else:
            return open(self.dictionary, 'rb')

    def load_userdict(self, f):
        '''
        Load personalized dict to improve detect rate.

        Parameter:
            - f : A plain text file contains words and their ocurrences.
                  Can be a file-like object, or the path of the dictionary file,
                  whose encoding must be utf-8.

        Structure of dict file:
        word1 freq1 word_type1
        word2 freq2 word_type2
        ...
        Word type may be ignored
        '''
        self.check_initialized()
        if isinstance(f, string_types):
            f_name = f
            f = open(f, 'rb')
        else:
            f_name = resolve_filename(f)
        for lineno, ln in enumerate(f, 1):
            line = ln.strip()
            if not isinstance(line, text_type):
                try:
                    line = line.decode('utf-8').lstrip('\ufeff')
                except UnicodeDecodeError:
                    raise ValueError('dictionary file %s must be utf-8' % f_name)
            if not line:
                continue
            # match won't be None because there's at least one character
            word, freq, tag = re_userdict.match(line).groups()
            if freq is not None:
                freq = freq.strip()
            if tag is not None:
                tag = tag.strip()
            self.add_word(word, freq, tag)

    def add_word(self, word, freq=None, tag=None):
        """
        Add a word to dictionary.

        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.
        """
        self.check_initialized()
        word = strdecode(word)
        freq = int(freq) if freq is not None else self.suggest_freq(word, False)
        self.FREQ[word] = freq
        self.total += freq
        if tag:
            self.user_word_tag_tab[word] = tag
        for ch in xrange(len(word)):
            wfrag = word[:ch + 1]
            if wfrag not in self.FREQ:
                self.FREQ[wfrag] = 0
        if freq == 0:
            finalseg.add_force_split(word)

    def del_word(self, word):
        """
        Convenient function for deleting a word.
        """
        self.add_word(word, 0)

    def suggest_freq(self, segment, tune=False):
        """
        Suggest word frequency to force the characters in a word to be
        joined or splitted.

        Parameter:
            - segment : The segments that the word is expected to be cut into,
                        If the word should be treated as a whole, use a str.
            - tune : If True, tune the word frequency.

        Note that HMM may affect the final result. If the result doesn't change,
        set HMM=False.
        """
        self.check_initialized()
        ftotal = float(self.total)
        freq = 1
        if isinstance(segment, string_types):
            word = segment
            for seg in self.cut(word, HMM=False):
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = max(int(freq * self.total) + 1, self.FREQ.get(word, 1))
        else:
            segment = tuple(map(strdecode, segment))
            word = ''.join(segment)
            for seg in segment:
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = min(int(freq * self.total), self.FREQ.get(word, 0))
        if tune:
            add_word(word, freq)
        return freq

    def tokenize(self, unicode_sentence, mode="default", HMM=True):
        """
        Tokenize a sentence and yields tuples of (word, start, end)

        Parameter:
            - sentence: the str(unicode) to be segmented.
            - mode: "default" or "search", "search" is for finer segmentation.
            - HMM: whether to use the Hidden Markov Model.
        """
        if not isinstance(unicode_sentence, text_type):
            raise ValueError("jieba: the input parameter should be unicode.")
        start = 0
        if mode == 'default':
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                yield (w, start, start + width)
                start += width
        else:
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                if len(w) > 2:
                    for i in xrange(len(w) - 1):
                        gram2 = w[i:i + 2]
                        if self.FREQ.get(gram2):
                            yield (gram2, start + i, start + i + 2)
                if len(w) > 3:
                    for i in xrange(len(w) - 2):
                        gram3 = w[i:i + 3]
                        if self.FREQ.get(gram3):
                            yield (gram3, start + i, start + i + 3)
                yield (w, start, start + width)
                start += width

    def set_dictionary(self, dictionary_path):
        with self.lock:
            abs_path = _get_abs_path(dictionary_path)
            if not os.path.isfile(abs_path):
                raise Exception("jieba: file does not exist: " + abs_path)
            self.dictionary = abs_path
            self.initialized = False


# default Tokenizer instance

dt = Tokenizer()

# global functions

get_FREQ = lambda k, d=None: dt.FREQ.get(k, d)
add_word = dt.add_word
calc = dt.calc
cut = dt.cut
lcut = dt.lcut
cut_for_search = dt.cut_for_search
lcut_for_search = dt.lcut_for_search
del_word = dt.del_word
get_DAG = dt.get_DAG
get_dict_file = dt.get_dict_file
initialize = dt.initialize
load_userdict = dt.load_userdict
set_dictionary = dt.set_dictionary
suggest_freq = dt.suggest_freq
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut(s):
    return dt._lcut(s)


def _lcut_no_hmm(s):
    return dt._lcut_no_hmm(s)


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut_for_search(s):
    return dt._lcut_for_search(s)


def _lcut_for_search_no_hmm(s):
    return dt._lcut_for_search_no_hmm(s)


def _pcut(sentence, cut_all=False, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if cut_all:
        result = pool.map(_lcut_all, parts)
    elif HMM:
        result = pool.map(_lcut, parts)
    else:
        result = pool.map(_lcut_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def _pcut_for_search(sentence, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if HMM:
        result = pool.map(_lcut_for_search, parts)
    else:
        result = pool.map(_lcut_for_search_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def enable_parallel(processnum=None):
    """
    Change the module's `cut` and `cut_for_search` functions to the
    parallel version.

    Note that this only works using dt, custom Tokenizer
    instances are not supported.
    """
    global pool, dt, cut, cut_for_search
    from multiprocessing import cpu_count
    if os.name == 'nt':
        raise NotImplementedError(
            "jieba: parallel mode only supports posix system")
    else:
        from multiprocessing import Pool
    dt.check_initialized()
    if processnum is None:
        processnum = cpu_count()
    pool = Pool(processnum)
    cut = _pcut
    cut_for_search = _pcut_for_search


def disable_parallel():
    global pool, dt, cut, cut_for_search
    if pool:
        pool.close()
        pool = None
    cut = dt.cut
    cut_for_search = dt.cut_for_search
