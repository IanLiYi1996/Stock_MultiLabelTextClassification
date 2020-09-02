import re
from zhconv import convert

from lxml import etree
import re


class DataClean(object):
    """
    来的原始数据都要经过这里进行处理
    """

    @staticmethod
    def fullwidth_to_halfwidth(ustring):
        """
        全角转半角
        :param ustring: string
        :return: string
        """
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        return rstring

    def clean_text(self, text):
        """
        清洗入口
        :param text: string
        :return: string
        """
        if not text:
            return ""
        if not isinstance(text, str):
            text = str(text)
        if not text.strip():
            return ""
        clean_html = etree.HTML(text=text).xpath('string(.)')
        simplified_text = convert(clean_html, "zh-cn")
        return self.fullwidth_to_halfwidth(simplified_text).lower()

class TextProcess(object):
    """
    文本数据的预处理，雪球
    根据自己情况来选择，预处理类型,目前还没有写好
    """

    def __init__(self, stock=True, url=True, time=True, money=True, number=True,
                 expression=True, user=True, percent=True, **kwargs):

        self.TEXT_CLEAN_PIPLINE = [
            {
                "pattern": re.compile(r"&nbsp;"),
                "replace": "SPACE",
                "enable": False
            },
            {
                # HTML 标签正则
                "pattern": re.compile(r'&[a-z0-9]+;|'
                                      r'<[a-z/][^>]*>|'
                                      r'http[^ <>]+|'
                                      r'<.*?>'),
                "replace": "_HTML_",
                "enable": False
            },
            {
                "pattern": re.compile(r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|'
                                      r'[a-z0-9.\-]+[.](?:com|net|org|edu|gov|'
                                      r'mil|aero|asia|biz|cat|coop|info|int|jobs|'
                                      r'mobi|museum|name|post|pro|tel|travel|xxx|'
                                      r'ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|'
                                      r'at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|'
                                      r'bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|'
                                      r'cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|'
                                      r'cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|'
                                      r'eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|'
                                      r'gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|'
                                      r'gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|'
                                      r'il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|'
                                      r'kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|'
                                      r'li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|'
                                      r'mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|'
                                      r'mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|'
                                      r'nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|'
                                      r'pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|'
                                      r'sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|'
                                      r'sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|'
                                      r'tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|'
                                      r'ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|'
                                      r'ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\['
                                      r'\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|'
                                      r'\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)'
                                      r'[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};'
                                      r':\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:'
                                      r'[.\-][a-z0-9]+)*[.](?:com|net|org|edu|'
                                      r'gov|mil|aero|asia|biz|cat|coop|info|int|'
                                      r'jobs|mobi|museum|name|post|pro|tel|'
                                      r'travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|'
                                      r'ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|'
                                      r'bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|'
                                      r'by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|'
                                      r'co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|'
                                      r'do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|'
                                      r'fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|'
                                      r'gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|'
                                      r'ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|'
                                      r'jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|'
                                      r'kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|'
                                      r'mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|'
                                      r'ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|'
                                      r'ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|'
                                      r'pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|'
                                      r'ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|'
                                      r'sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|'
                                      r'td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|'
                                      r'tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|'
                                      r'vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?'
                                      r'!@)))'),
                "replace": "",
                "enable": url
            },
            {
                # 金@钱的部分正则
                "pattern": re.compile(r"\[¥([.0-9]+)\]"),
                "replace": "Money",
                "enable": money
            },
            {
                # 表情
                "pattern": re.compile(r"(?:\[仰慕\]|\[韭菜\]|\[捂脸\]|\[减\]|"
                                      r"\[生气\]|\[困\]|\[难过\]|\[秘密\]|"
                                      r"\[能力圈\]|\[中签\]|\[无语\]|\[不赞\]|"
                                      r"\[多\]|\[围观\]|\[诅咒\]|\[笑\]|"
                                      r"\[抄底\]|\[停\]|\[困惑\]|\[干杯\]|"
                                      r"\[大笑\]|\[好逊\]|\[俏皮\]|\[尴尬\]|"
                                      r"\[凋谢\]|\[加仓\]|\[可爱\]|\[调皮\]|"
                                      r"\[哭泣\]|\[主力\]|\[鼓鼓掌\]|\[护城河\]|"
                                      r"\[毛估估\]|\[加油\]|\[赞\]|\[滴汗\]|"
                                      r"\[抠鼻\]|\[晕\]|\[关灯吃面\]|\[卖出\]|"
                                      r"\[哈哈\]|\[加\]|\[傲慢\]|\[空仓\]|"
                                      r"\[贬\]|\[挣扎\]|\[心心\]|\[涨\]|"
                                      r"\[看多\]|\[屎\]|\[卖\]|\[献花花\]|"
                                      r"\[握手\]|\[怒了\]|\[梭哈\]|\[闭嘴\]|"
                                      r"\[讨厌\]|\[跪了\]|\[买\]|\[满仓\]|"
                                      r"\[心碎了\]|\[吐血\]|\[傲\]|\[汗\]|"
                                      r"\[复盘\]|\[买入\]|\[呵呵傻逼\]|\[哭\]|"
                                      r"\[赚大了\]|\[贪财\]|\[害羞\]|\[笑哭\]|"
                                      r"\[摊手\]|\[很赞\]|\[成交\]|\[不说了\]|"
                                      r"\[空\]|\[疑问\]|\[赞成\]|\[亲亲\]|"
                                      r"\[跌\]|\[微笑\]|\[看空\]|\[割肉\]|"
                                      r"\[可怜\]|\[心碎\]|\[吐舌\]|\[献花\]|"
                                      r"\[色\]|\[减仓\]|\[一坨屎\]|\[好困惑\]|"
                                      r"\[好失望\]|\[失望\]|\[卖身\]|\[胜利\]|"
                                      r"\[鼓掌\]|\[国家队\]|\[不屑\]|\[爱\]|"
                                      r"\[跳水\]|\[不知道\]|\[困顿\]|\[呵呵\]|"
                                      r"\[牛\]|\[为什么\]|\[想一下\]|\[亏大了\]|"
                                      r"\[囧\]|\[思考\])"),
                "replace": "EXP",
                "enable": expression
            },
            {
                # @用户正则
                "pattern": re.compile(r"[@＠]([\u4E00-\u9FFFa-zA-Z0-9_-]{2,})"),
                "replace": "U",
                "enable": user
            },
            {
                # 匹配时间格式问题
                "pattern": re.compile(
                    r"[0-9]{4}[0-9]{2}[0-9]{2}[0-9]{2}[0-9]{2}[0-9]{2}|"
                    r"[0-9]{4}[0-9]{2}[0-9]{2}|"
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}|"
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2}|"
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}"
                    r"(([0-9]|一|二|三|四|五|六|七|八|九|十)+[ ]*(年|月|日|季度|年度|月份|小时|分钟|刻钟|刻|秒)+)+|"
                    r"[0-9]+(:|：)[0-9]+|"
                    r"[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}|"
                    r"(本|上|下)?(周|星期)(一|二|三|四|五|六|日|末)"
                ),
                "replace": "T",
                "enable": time
            },
            {
                "pattern": re.compile(r"(\d+)%|(\-|\+)\d+(\.\d+)%?"),
                "replace": "P",
                "enable": percent
            },
            {
                "pattern": re.compile(r"[\+\-]?[0-9]+\.[0-9]+|"  # 小数
                                      r"[\+\-]?[0-9]+%|"  # 百分数
                                      r"[\+\-]?[0-9]+|"  # 整数
                                      r"[0-9]{7,}|"  # 啥？
                                      r"[,0-9\.]+(百|千|万|亿|元|块)|"  # 钱
                                      # 中文百分数
                                      r"(百分之|千分之|万分之)([0-9]|一|二|三|四|五|六|七|八|九|十)+|"
                                      r"￥[0-9\.]+",  # 人民币
                                      re.X | re.M),
                "replace": "N",
                "enable": number
            },
            {
                "pattern": re.compile(r'\$\S+(\S+)\$'),
                "replace": "S",
                "enable": stock
            }
        ]
        self.special_compile = re.compile(
            r"\$\S+\$看我主|^我(刚刚关注|刚刚调整)了雪球(组合|话题)|"
            r"^我刚(打赏|发现)了这个|我发现这个(实盘|雪球实盘|雪球)组合|^私募工场\(|"
            r"^我刚刚在#雪球实盘交易|快来看看|关注此策略即可收到最新消息推送，请在 App 内点击查看|"
            r"^#雪球翻翻卡#|"
            r"^新讨论"
        )

    def extract_comment(self, text):
        """
        提取status中的comment中真正的内容,使用这个必须是经过data_clean后的数据
        :param text:
        :return:
        """
        if not text or not isinstance(text, str):
            return text
        comment = text.split("//@")[0].split(":", 1)[-1]
        if comment:
            return comment.strip()
        return comment

    def is_special_page(self, text):
        """
        todo:这部分可以修改
        处理的对象如下
        <p>$海川智能(SZ300720)$看我主页，雪球群每天知识分享，好不错。一起交流</p>
        <p>$金陵体育(SZ300651)$看我主页，这个交流大会不错，球友一起沟通</p>
        我刚刚关注了雪球组合 $已关停(ZH006428)$ ，当前净值45.7845。
        我刚刚调整了雪球组合 $鲤鱼事务所实盘一(ZH937056)$  的仓位。
        新讨论 2019-10-16 12-04-14
        我刚刚关注了雪球话题 #今天是几号# 欢迎一起来讨论
        #雪球翻翻卡# 【爱康国宾体验折扣卡】get√，终身VIP会员卡、购物优惠券、医疗险重疾险有趣又有好礼，每天都有机会翻出惊喜 https://xueqiu.com/growth-activity/flip-card?
        $特变电工(SH600089)$看我主页，我加入了雪球球友大会，一起交流
        我刚刚抢到了%
        我刚分给这个回答%
        我刚打赏了这个帖子%
        我刚打赏了这篇帖子%
        我在雪球创建了一个%
        我在雪球加入了%
        我刚发现了%
        私募工场(%
        我发现这个实盘组合%
        我发现这个雪球实盘组合%
        我发现这个雪球组合%
        %最新价%涨跌幅%
        我刚刚在#雪球实盘交易#%
        快来看看%
        %关注此策略即可收到最新消息推送，请在 App 内点击查看%
        %以%买入% (source = 持仓盈亏)
        Args:
            text:
        Returns:
        """
        if re.search(self.special_compile, text):
            return True
        return False

    def process_text(self, text):
        """
        文本处理
        :param text:
        :return:
        """
        if not text:
            return ""
        if self.is_special_page(text):
            return "__special__"
        for single_regex in self.TEXT_CLEAN_PIPLINE:
            if not single_regex["enable"]:
                continue
            text = re.sub(single_regex["pattern"], single_regex["replace"],
                          text)

        return text