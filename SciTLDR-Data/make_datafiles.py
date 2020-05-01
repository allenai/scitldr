import sys
import os
import hashlib
import struct
import subprocess
import collections
import argparse
import re


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding="utf-8") as f:
    for line in f:
      lines.append(line.strip())
  # import ipdb; ipdb.set_trace()
  print(text_file)
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode())
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]

def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string
  abstract = ' '.join(highlights)

  return article, abstract


def write_to_bin(url_file, stories_dir, out_prefix):
  """Reads the .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print("Making bin file for URLs listed in %s..." % url_file)
  url_list = read_text_file(url_file)
  # url_hashes = get_url_hashes(url_list)
  story_fnames = [s+".story" for s in url_list]
  num_stories = len(story_fnames)

  with open(out_prefix + '.source', 'wt', encoding="utf-8") as source_file, open(out_prefix + '.target', 'wt', encoding="utf-8") as target_file:
    for idx,s in enumerate(story_fnames):
      if idx % 1000 == 0:
        print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

      # Look in the story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(stories_dir, s)):
        story_file = os.path.join(stories_dir, s)
      else:
        print("Error: Couldn't find story file %s in either story directory %s" % (s, stories_dir))
      # Get the strings to write to .bin file
      article, abstract = get_art_abs(story_file)

      # Write article and abstract to files
      source_file.write(article + '\n')
      target_file.write(abstract + '\n')

  print("Finished writing files")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--stories_dir', default='url_lists')
  parser.add_argument('--urldir', default='mapping')
  parser.add_argument('--finished_files_dir', default='data')
  
  args = parser.parse_args()

  all_train_urls = os.path.join(args.urldir, "mapping_train.txt")
  all_val_urls = os.path.join(args.urldir, "mapping_valid.txt")
  all_test_urls = os.path.join(args.urldir, "mapping_test.txt")

  # Create some new directories
  if not os.path.exists(args.finished_files_dir): 
    os.makedirs(args.finished_files_dir)

  # Read the stories, do a little postprocessing then write to bin files
  write_to_bin(all_test_urls, args.stories_dir, os.path.join(args.finished_files_dir, "test"))
  write_to_bin(all_val_urls, args.stories_dir, os.path.join(args.finished_files_dir, "val"))
  write_to_bin(all_train_urls, args.stories_dir, os.path.join(args.finished_files_dir, "train"))
