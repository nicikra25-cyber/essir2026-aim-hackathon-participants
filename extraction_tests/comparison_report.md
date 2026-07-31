# PDF Extraction Comparison

This report compares extraction quality on PDF pages 2, 9, 16, 25, 27. The automatic indicators are preliminary and must be checked against the visible PDF.

## Summary

| Extractor | Important phrases found | Broken-word indicators | Damaged symbols | Tables detected | Preliminary reason |
|---|---:|---:|---:|---:|---|
| pypdf | 3/3 | 10 | 0 | 0 | preserved all tested Level-1 phrases; 10 possible broken words |
| PyMuPDF | 3/3 | 1 | 0 | 0 | preserved all tested Level-1 phrases; 1 possible broken words |
| pdfplumber | 3/3 | 18 | 0 | 0 | preserved all tested Level-1 phrases; 18 possible broken words |

## Important phrase checks

### Target phrase: `threat of new entrants`

**pypdf: found on PDF page(s) [2, 25].**

> and the formation and effectiveness of strate- gies within organisations. Porter (1979) famously claimed that five forces shape the competitive environment of an organisation: the threat of new entrants, the bargain - ing power of suppliers, the bargaining power of buyers, rivalry among existing com- petitors, and the threat of substitute products or services. These five forces c

**PyMuPDF: found on PDF page(s) [2, 25].**

> and the formation and effectiveness of strate­ gies within organisations. Porter (1979) famously claimed that five forces shape the competitive environment of an organisation: the threat of new entrants, the bargain­ ing power of suppliers, the bargaining power of buyers, rivalry among existing com­ petitors, and the threat of substitute products or services. These five forces co

**pdfplumber: found on PDF page(s) [2, 25].**

> and the formation and effectiveness of strate- gies within organisations. Porter (1979) famously claimed that five forces shape the competitive environment of an organisation: the threat of new entrants, the bargain- ing power of suppliers, the bargaining power of buyers, rivalry among existing com- petitors, and the threat of substitute products or services. These five forces co

### Target phrase: `organisational performance and competitive advantage`

**pypdf: phrase not found exactly.**

**PyMuPDF: phrase not found exactly.**

**pdfplumber: phrase not found exactly.**

### Target phrase: `505 complete responses were available for data analysis`

**pypdf: found on PDF page(s) [9].**

> s, 548 completed questionnaires were received, a response rate of 47.24%. However, as speeder checks removed 10 of these and a further 33 were removed due to repetitive responses, 505 complete responses were available for data analysis (i.e. a response rate of 43.53%). Table 1 shows that the majority of the responding organisations had 100–499 employees (52.9%), 135 (26.8%) had 500–999 employees, 78 (15.5%) had 

**PyMuPDF: found on PDF page(s) [9].**

> s, 548 completed questionnaires were received, a response rate of 47.24%. However, as speeder checks removed 10 of these and a further 33 were removed due to repetitive responses, 505 complete responses were available for data analysis (i.e. a response rate of 43.53%). Table 1 shows that the majority of the responding organisations had 100–499 employees (52.9%), 135 (26.8%) had 500–999 employees, 78 (15.5%) had 

**pdfplumber: found on PDF page(s) [9].**

> s, 548 completed questionnaires were received, a response rate of 47.24%. However, as speeder checks removed 10 of these and a further 33 were removed due to repetitive responses, 505 complete responses were available for data analysis (i.e. a response rate of 43.53%). Table 1 shows that the majority of the responding organisations had 100–499 employees (52.9%), 135 (26.8%) had 500–999 employees, 78 (15.5%) had 


## Side-by-side page examples

### PDF page 2

#### pypdf

- Characters: 3589
- Possible broken words: 4
- Very short lines: 3
- Important phrases found: 1

Example:

> K. Baird et al. 1 Introduction Michael Porter’s (1979) ‘How competitive forces shape strategy’ has influenced the field of research on competitive forces and the formation and effectiveness of strate- gies within organisations. Porter (1979) famously claimed that five forces shape the competitive environment of an organisation: the threat of new entrants, the bargain - ing power of suppliers, the bargaining power of buyers, rivalry among existing com- petitors, and the threat of substitute products or services. These five forces combine to shape the industry structure and influence the way organisations compete within an industry, with organisations tending to emphasise Porter’s (1985) cost  ...

#### PyMuPDF

- Characters: 3657
- Possible broken words: 1
- Very short lines: 1
- Important phrases found: 1

Example:

> 304 K. Baird et al. 1 Introduction Michael Porter’s (1979) ‘How competitive forces shape strategy’ has influenced the field of research on competitive forces and the formation and effectiveness of strate­ gies within organisations. Porter (1979) famously claimed that five forces shape the competitive environment of an organisation: the threat of new entrants, the bargain­ ing power of suppliers, the bargaining power of buyers, rivalry among existing com­ petitors, and the threat of substitute products or services. These five forces combine to shape the industry structure and influence the way organisations compete within an industry, with organisations tending to emphasise Porter’s (1985) co ...

#### pdfplumber

- Characters: 3549
- Possible broken words: 9
- Very short lines: 2
- Important phrases found: 1

Example:

> 304 K. Baird et al. 1 Introduction Michael Porter’s (1979) ‘How competitive forces shape strategy’ has influenced the field of research on competitive forces and the formation and effectiveness of strate- gies within organisations. Porter (1979) famously claimed that five forces shape the competitive environment of an organisation: the threat of new entrants, the bargain- ing power of suppliers, the bargaining power of buyers, rivalry among existing com- petitors, and the threat of substitute products or services. These five forces combine to shape the industry structure and influence the way organisations compete within an industry, with organisations tending to emphasise Porter’s (1985) co ...

**pdfplumber tables detected on this page:** 0

### PDF page 9

#### pypdf

- Characters: 3272
- Possible broken words: 2
- Very short lines: 4
- Important phrases found: 1

Example:

> The effect of Porter’s competitive forces on competitive advantage and… (Langfied-Smith et al., 2018), thereby enabling managers to respond to and cope with the intensity of competitive forces more effectively. Therefore, we argue that traditional management accounting practices will play a positive moderating role, enhancing an organisations’ ability to cope with the inten - sity of competitive forces in a way which can enhance both their competitiveness (i.e. competitive advantage) and profitability. Specifically, it is expected that there will be a stronger (weaker) association between the intensity of competitive forces with organisational outcomes (competitive advantage and organisation ...

#### PyMuPDF

- Characters: 3309
- Possible broken words: 0
- Very short lines: 3
- Important phrases found: 1

Example:

> The effect of Porter’s competitive forces on competitive advantage and… 311 (Langfied-Smith et al., 2018), thereby enabling managers to respond to and cope with the intensity of competitive forces more effectively. Therefore, we argue that traditional management accounting practices will play a positive moderating role, enhancing an organisations’ ability to cope with the inten­ sity of competitive forces in a way which can enhance both their competitiveness (i.e. competitive advantage) and profitability. Specifically, it is expected that there will be a stronger (weaker) association between the intensity of competitive forces with organisational outcomes (competitive advantage and organisat ...

#### pdfplumber

- Characters: 3238
- Possible broken words: 6
- Very short lines: 3
- Important phrases found: 1

Example:

> The effect of Porter’s competitive forces on competitive advantage and… 311 (Langfied-Smith et al., 2018), thereby enabling managers to respond to and cope with the intensity of competitive forces more effectively. Therefore, we argue that traditional management accounting practices will play a positive moderating role, enhancing an organisations’ ability to cope with the inten- sity of competitive forces in a way which can enhance both their competitiveness (i.e. competitive advantage) and profitability. Specifically, it is expected that there will be a stronger (weaker) association between the intensity of competitive forces with organisational outcomes (competitive advantage and organisat ...

**pdfplumber tables detected on this page:** 0

### PDF page 16

#### pypdf

- Characters: 690
- Possible broken words: 3
- Very short lines: 15
- Important phrases found: 0

Example:

> K. Baird et al. Table 4 Correlations and square root of A VE scores Intensity of com- petitive forces Organisational performance Competitive advantage Contemporary MAPs Traditional MAPs Differentiation strategy Low- cost strat- egy Intensity of competitive forces 0.854 Organisational performance 0.543 0.761 Competitive advantage 0.577 0.681 0.736 Contemporary MAPs 0.618 0.576 0.630 0.735 Traditional MAPs 0.622 0.569 0.605 0.685 0.721 Differentiation strategy 0.676 0.535 0.604 0.692 0.727 0.731 Low-cost strategy 0.663 0.507 0.563 0.641 0.703 0.729 0.762 NB The diagonal figures in bold represent the square root of the average variance extracted scores for each construct 1 3 318

#### PyMuPDF

- Characters: 4261
- Possible broken words: 0
- Very short lines: 6
- Important phrases found: 0

Example:

> 318 K. Baird et al. Low-coststrat­egy 0.762 Differentiationstrategy 0.731 0.729 TraditionalMAPs 0.721 0.727 0.703 ContemporaryMAPs 0.735 0.685 0.692 0.641 construct each for scores Competitiveadvantage 0.736 0.630 0.605 0.604 0.563 extracted variance average Organisationalperformance 0.761 0.681 0.576 0.569 0.535 0.507 the of com­ root of forces square scores the Intensitypetitive 0.854 0.543 0.577 0.618 0.622 0.676 0.663 AVE of represent root bold square in forces and figures performance MAPs strategy competitive advantage MAPs of strategy diagonal Correlations 4 The Table Intensity Organisational Competitive Contemporary Traditional Differentiation Low-cost NB 1 3

#### pdfplumber

- Characters: 683
- Possible broken words: 0
- Very short lines: 91
- Important phrases found: 0

Example:

> K. Baird et al. serocs EVA fo toor erauqs dna snoitalerroC 4 elbaT -woL noitaitnereffiD lanoitidarT yraropmetnoC evititepmoC lanoitasinagrO -moc fo ytisnetnI tsoc ygetarts sPAM sPAM egatnavda ecnamrofrep secrof evititep -tarts yge 458.0 secrof evititepmoc fo ytisnetnI 167.0 345.0 ecnamrofrep lanoitasinagrO 637.0 186.0 775.0 egatnavda evititepmoC 537.0 036.0 675.0 816.0 sPAM yraropmetnoC 127.0 586.0 506.0 965.0 226.0 sPAM lanoitidarT 137.0 727.0 296.0 406.0 535.0 676.0 ygetarts noitaitnereffiD 267.0 927.0 307.0 146.0 365.0 705.0 366.0 ygetarts tsoc-woL tcurtsnoc hcae rof serocs detcartxe ecnairav egareva eht fo toor erauqs eht tneserper dlob ni serugfi lanogaid ehT BN 318 1 3

**pdfplumber tables detected on this page:** 0

### PDF page 25

#### pypdf

- Characters: 1944
- Possible broken words: 1
- Very short lines: 4
- Important phrases found: 1

Example:

> The effect of Porter’s competitive forces on competitive advantage and… Rivalry among existing competitors There is a large number of competitors in our industry. There are no clear leaders in our market† (0.777). The market is growing quickly. It is easy to compete in the industry because of low fixed costs† (0.772). Competition is high, as our products (goods and/or services) are standardised† (0.743). Bargaining power of buyers We have only a few customers, and hence, losing one is critical to our success. We have a limited number of customers whose purchases represent a large propor- tion of our total revenue from goods and/or services† (0.820). Our customers have a detailed knowledge of ...

#### PyMuPDF

- Characters: 2037
- Possible broken words: 0
- Very short lines: 3
- Important phrases found: 1

Example:

> The effect of Porter’s competitive forces on competitive advantage and… 327 Rivalry among existing competitors There is a large number of competitors in our industry. There are no clear leaders in our market† (0.777). The market is growing quickly. It is easy to compete in the industry because of low fixed costs† (0.772). Competition is high, as our products (goods and/or services) are standardised† (0.743). Bargaining power of buyers We have only a few customers, and hence, losing one is critical to our success. We have a limited number of customers whose purchases represent a large propor­ tion of our total revenue from goods and/or services† (0.820). Our customers have a detailed knowledg ...

#### pdfplumber

- Characters: 1935
- Possible broken words: 3
- Very short lines: 3
- Important phrases found: 1

Example:

> The effect of Porter’s competitive forces on competitive advantage and… 327 Rivalry among existing competitors There is a large number of competitors in our industry. There are no clear leaders in our market† (0.777). The market is growing quickly. It is easy to compete in the industry because of low fixed costs† (0.772). Competition is high, as our products (goods and/or services) are standardised† (0.743). Bargaining power of buyers We have only a few customers, and hence, losing one is critical to our success. We have a limited number of customers whose purchases represent a large propor- tion of our total revenue from goods and/or services† (0.820). Our customers have a detailed knowledg ...

**pdfplumber tables detected on this page:** 0

### PDF page 27

#### pypdf

- Characters: 2598
- Possible broken words: 0
- Very short lines: 5
- Important phrases found: 0

Example:

> The effect of Porter’s competitive forces on competitive advantage and… Finding ways to reduce costs. The level of operating efficiency. The level of production capacity utilization† (0.768). Price competition† (0.738). Product-differentiation strategy Using new methods and technologies to create superior products† (0.672). New product development† (0.724). The rate of new product introduction to the market. The number of new products offered to the market† (0.761). Intensity of advertising and marketing. Developing and utilizing the sales force† (0.763). Building a strong brand identification. Acknowledgements This work was supported by the Institute of Management Accountants (IMA) Research ...

#### PyMuPDF

- Characters: 2695
- Possible broken words: 0
- Very short lines: 4
- Important phrases found: 0

Example:

> The effect of Porter’s competitive forces on competitive advantage and… 329 Finding ways to reduce costs. The level of operating efficiency. The level of production capacity utilization† (0.768). Price competition† (0.738). Product-differentiation strategy Using new methods and technologies to create superior products† (0.672). New product development† (0.724). The rate of new product introduction to the market. The number of new products offered to the market† (0.761). Intensity of advertising and marketing. Developing and utilizing the sales force† (0.763). Building a strong brand identification. Acknowledgements This work was supported by the Institute of Management Accountants (IMA) Rese ...

#### pdfplumber

- Characters: 2581
- Possible broken words: 0
- Very short lines: 4
- Important phrases found: 0

Example:

> The effect of Porter’s competitive forces on competitive advantage and… 329 Finding ways to reduce costs. The level of operating efficiency. The level of production capacity utilization† (0.768). Price competition† (0.738). Product-differentiation strategy Using new methods and technologies to create superior products† (0.672). New product development† (0.724). The rate of new product introduction to the market. The number of new products offered to the market† (0.761). Intensity of advertising and marketing. Developing and utilizing the sales force† (0.763). Building a strong brand identification. Acknowledgements This work was supported by the Institute of Management Accountants (IMA) Rese ...

**pdfplumber tables detected on this page:** 0

## Manual observations

Complete these points after comparing the examples with the PDF:

- Were the columns mixed?
- Were sentences placed in the correct order?
- Were words broken across lines?
- Were headings and footnotes mixed with the main text?
- Were tables readable?
- Which extractor preserved the Level-1 evidence most exactly?

## Final decision

**Selected extractor:** 

**Why it was selected:** 

**Example showing the improvement:** 
