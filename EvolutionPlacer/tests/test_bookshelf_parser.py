import sys,os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import bin.parser.bookshelf_parser as parser

x = parser.BookshelfParser()
