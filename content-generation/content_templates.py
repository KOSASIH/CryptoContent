"""
Content Templates

This module provides templates for different content formats.
"""

from typing import Dict

class ContentTemplate:
    def __init__(self, template: str, variables: Dict[str, str]):
        """
        Initialize a content template.

        Args:
            template (str): The template string.
            variables (Dict[str, str]): The variables to replace in the template.
        """
        self.template = template
        self.variables = variables

    def format(self, words: List[str]) -> str:
        """
        Format the template with the given words.

        Args:
            words (List[str]): The words to insert into the template.

        Returns:
            str: The formatted content.
        """
        # Replace the variables in the template with the given words
        content = self.template
        for variable, word in zip(self.variables.keys(), words):
            content = content.replace("{" + variable + "}", word)

        return content

class ArticleTemplate(ContentTemplate):
    def __init__(self):
        super().__init__(
            template="The {topic} of {year} has been a wild ride. {sentence1} {sentence2} {sentence3}.",
            variables={"topic": "", "year": "", "sentence1": "", "sentence2": "", "sentence3": ""},
        )

class SocialMediaPostTemplate(ContentTemplate):
    def __init__(self):
        super().__init__(
            template="Did you know that {fact}? {hashtag} #cryptocurrency",
            variables={"fact": "", "hashtag": ""},
        )

class BlogPostTemplate(ContentTemplate):
    def __init__(self):
        super().__init__(
            template="The future of {industry} is {adjective}. {paragraph1} {paragraph2} {paragraph3}.",
            variables={"industry": "", "adjective": "", "paragraph1": "", "paragraph2": "", "paragraph3": ""},
        )
