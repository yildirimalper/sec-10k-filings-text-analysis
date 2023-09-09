from sec_api import QueryApi
import json

queryApi = QueryApi(api_key="c0783bfe7219bc8ef761141c0d7dc0632f83ad15c8dac41375ced6399c41d19b")

base_query = {
  "query": { 
      "query_string": { 
          "query": "PLACEHOLDER", # will be replaced with the search universe
          "time_zone": "America/New_York"
      } 
  },
  "from": "0",
  "size": "200",
  # sort returned filings by the filedAt key/value
  "sort": [{ "filedAt": { "order": "desc" } }]
}

# open the file to store the 10-k filing URLs
# "a" : open for writing, appending to the end of file if it exists 
log_file = open("filing_urls.txt", "a")


# download 10-K filings URLs for the years 2020 and 2021:
# --------------------------------------------------
for year in range(2022, 2019, -1):
  print("Starting download for year {year}".format(year=year))
  
  # a single search universe is represented as a month of the given year
  for month in range(1, 13, 1):
    # set search universe for year-month combination
    universe_query = \
        "formType:(\"10-K\") AND " + \
        "filedAt:[{year}-{month:02d}-01 TO {year}-{month:02d}-31]" \
        .format(year=year, month=month)
  
    # set new query universe for year-month combination
    base_query["query"]["query_string"]["query"] = universe_query;

    # fetch all filings for the given year-month combination
    for from_batch in range(0, 400, 200):
      # set new "from" starting position of search 
      base_query["from"] = from_batch;

      response = queryApi.get_filings(base_query)

      # no more filings in search universe
      if len(response["filings"]) == 0:
        break;

      # for each filing, only save the URL pointing to the filing itself and ignore all other data. 
      # the URL is set in the dict key "linkToFilingDetails"
      urls_list = list(map(lambda x: x["linkToFilingDetails"], response["filings"]))

      # transform list of URLs into one string by joining all list elements
      urls_string = "\n".join(urls_list) + "\n"
      
      log_file.write(urls_string)

    print("Filing URLs downloaded for {year}-{month:02d}".format(year=year, month=month))

log_file.close()

print("All URLs downloaded")

# Thanks to @janlukasschroeder for the code
# More resources on this code:
# https://sec-api.io/
# https://medium.com/@jan_5421/how-to-download-and-scrape-10-k-filings-from-sec-edgar-b0d245fc8d48
# https://www.tutorialspoint.com/json/json_quick_guide.htm