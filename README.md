# Project Burbank

Code for the paper: "Predictive Analytics of Varieties of Russet Potatoes" 

## Goals of the Project

The main objective of the potato breeder is to develop processor grade potato varieties (French fry raw) with acceptable consumer attributes with reduced levels of asparagine and reducing sugars in the raw state to reduce the levels of acrylamide. The new varieties are preferred to have long term storage capability sufficient to substitute for the Russet Burbank variety. They must be non-genetically modified. They must have values that fall within the five-year regional average for commercially produced regional standard variety as defined below:

| Attributes (musts) | Target | Range |
|--------------------|--------|-------|
| Specific Gravity | 1.084 | 1.082-1.088 |
| % 6-oz Weight (>4 in length) | 70% | 65-75% |
| % 10-oz Weight| 30% | 25-40% |
| % >= USDA 3 | 0% | 0-2% |
| % Sugar Ends | 0% | 0-2% |
| Storability | 9 months | 9 months |
| Length to Width Ratio | 1.75 | 1.75-2.50 |
| Asparagine Level | 0.1g/100g |  |
| Reducing Sugar Level | 2.5mg/g|  |
| Total Glycoalkaloids (TGA) | <20 ppm |  |
| Internal Defects | 0% | 0-2% |
| Yield | ≥ to standard variety for the region. |  |


| Attributes (wants) | Target | Range |
|--------------------|--------|-------|
| Disease Resistance | Equal To or More Resistant | |
| Shatter Bruise     | Equal To or More Resistant | |
| Blackspot Bruise Free | 80% | Equal To or More Resistant |
| White Flesh | Equal to RB | |
| After cooking darkening resistance | | |

## Goals of the Data Analysis

We wish to automate the process of discriminating varieties that meet or exceed expected targets from those that do not meet the standards above. Currently the process is manual, where the potato breeding expert looks at every datum in order to make a decision. The goal is to write a classification algorithm/routine that outputs a wheter a particular variety should be kept or dropped from further trials, along with the probability of keeping the variety.

## Description of Data Provided

The potato breeder provided raw data for the Oregon state trials of the Russet potato for the years 2013 through 2021 in the form of Excel workbooks. For each year there is a book for each of 3-4 trial locations within Oregon, they are: Corvallis, Hermiston, Klamath Falls and Ontario for the years 2013 and 2014; and Hermiston (both Early and Late seasons in separate books), Klamath Falls and Ontario for the years 2015 through 2021.

There are 1086 total clones, 216 of which are unique. Years in trial range from 1-3. Most clones are in year 1 of trial. There are three controls measured for all regions and all years: Ranger Russet, Russet Burbank and Russet Norkotah; and 2 other controls available only for Hermiston (Early): Shepody and Hermiston (Late): Umatilla, measured for some but not all years.

For more information regarding the dataset and analyses, please read our paper.

## Running the Example

Required packages are listed in the file 'requirements.yml'. The python files have documentation. Create a new conda environment:

`conda env create -f requirements.yml`

For preparing the data, `cd` into the 'burbank' directory and run:
`python -m example --action='prepare'`

For training and testing on the prepared dataset run:
`python -m example --action='train'`

For predictions on the prepared dataset run:
`python -m example --action='predict'`