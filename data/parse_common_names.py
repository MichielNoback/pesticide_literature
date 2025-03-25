import re

pattern = re.compile(r"\[(?P<name>.+?)\]\((?P<url>.+?)\)")
in_file = "all_common_names.txt"
out_file = "common_names.csv"


out = open(out_file, "w")
out.write("name\turl\n")
with open(in_file, "r") as f:
    for i, line in enumerate(f):
        #[abamectin](http://www.bcpcpesticidecompendium.org/abamectin.html)  
        line = line.strip()
        if line:
            if pattern.match(line):
                match = pattern.match(line)
                name = match.group("name")
                if "\t" in name:
                    raise ValueError(f"Comma in name: {name}")
                url = match.group("url")
                out.write(f"{name}\t{url}\n")
                #print(f'name: {name}, url: {url}')
        # if i == 10:
        #     break


out.close()