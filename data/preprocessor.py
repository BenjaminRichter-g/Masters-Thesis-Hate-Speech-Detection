import re
from bs4 import BeautifulSoup
import emoji

class PreProcessor:
    def __init__(self):
        self.url_reg               = re.compile(r'https?://\s*[^\s<>"\'`]+')
        self.mention_reg           = re.compile(r'@\s*[\w\.-]+@[\w\.-]+')
        self.hashtag_reg           = re.compile(r'#\s*[\w-]+')
        self.emoji_placeholder_reg = re.compile(r':[^\s:]+:')

    def demojize_text(self, text):
        return emoji.demojize(text, language='en')

    def preprocess(self, html_content):
        if not html_content:
            return {}

        soup = BeautifulSoup(html_content, 'lxml')
        links_in_html = [a['href'] for a in soup.find_all('a', href=True)]
        raw_text = soup.get_text(separator=' ')

        demojized = self.demojize_text(raw_text)

        mentions     = [m.strip() for m in self.mention_reg.findall(demojized)]
        hashtags     = [h.strip() for h in self.hashtag_reg.findall(demojized)]
        links        = list({*links_in_html, *self.url_reg.findall(demojized)})
        emojis       = self.emoji_placeholder_reg.findall(demojized)
        emoji_names  = [e.strip(':').replace('_', ' ') for e in emojis]

        remove_pat = re.compile(
            f"({self.url_reg.pattern})|"
            f"({self.mention_reg.pattern})"
        )
        intermediate = remove_pat.sub('', demojized)

        clean_text = re.sub(r'\s+', ' ', intermediate).strip()

        return {
            "clean_text": clean_text,
            "mentions": mentions,
            "hashtags": hashtags,
            "links": sorted(links),
            "emojis": emojis,
            "emoji_names": emoji_names
        }    
