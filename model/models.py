from dataclasses import dataclass


@dataclass
class NextCloudObject:
    href: str
    last_modified: str
    resource_type: str
    content_length: str
    etag: str

    def __init__(self, **kwargs):
        self.href = kwargs.get('href')
        self.last_modified = kwargs.get('last_modified')
        self.resource_type = kwargs.get('resource_type')
        self.content_length = kwargs.get('content_length')
        self.etag = kwargs.get('etag')
