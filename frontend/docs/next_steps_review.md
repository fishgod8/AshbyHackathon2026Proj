# Suggested next steps: review sheet for a pathologist

**Context.** Our tool looks at one H&E photo of breast tissue (BACH: normal, benign, in situ carcinoma,
invasive carcinoma). When the AI cannot decide between two classes, it now suggests what to do next
instead of leading with papers. Everything below is a **draft** written only from verbatim quotes of
the cited papers. Nothing is shown as reviewed until you confirm it.

For each pair, please mark: **Agree** / **Change to …** / **Remove**, and tell us anything you would
order instead or in addition (we will add a citation).

Shown for every flagged pair, after any stain:
- Look at more of the lesion: other fields, or deeper levels from the block
- Ask a colleague for a second opinion

| # | AI is between | Draft next step | How to read it (draft) | Source (verbatim quote) | Your review |
|---|---|---|---|---|---|
| 1 | In situ carcinoma vs invasive carcinoma | Stain: p63 and 34βE12 | These help show whether the myoepithelial layer around the tumor nests is intact: intact supports in situ carcinoma, absent supports invasion. | Pinto et al., Pathobiology 2022 (PMID 35367980): "We found that p63 and 34βE12 show promise in aiding in the differential diagnosis between in situ and invasive lesions and that most other diagnostic markers may be used as in tissue." Rakha et al., Histopathology 2018 (PMID 29197112): "Differentiation of these two different lesions in breast pathological diagnosis is based typically on the presence of an intact barrier between the malignant epithelial cells and stroma; namely, the myoepithelial cell (MEC) layer and surrounding basement membrane (BM)." | |
| 2 | Benign lesion vs in situ carcinoma | Stain: CK5/6 | If the benign candidate is usual ductal hyperplasia: it usually co-expresses CK5/6, while atypical ductal hyperplasia and DCIS typically lack it. | Going et al., J Pathol 2003 (PMID 12474219): "Immunohistochemistry shows that most HUT have a mixed precursor phenotype resembling normal breast with co-expression of cytokeratin 5/6, 8/18/19 and SMA, in contrast to atypical ductal hyperplasia (ADH) and ductal carcinoma in situ which typically have a 'mature' luminal phenotype positive for cytokeratin 8/18/19 but lacking cytokeratin 5/6 expression." | |
| 3 | Normal tissue vs in situ carcinoma | Stain: CK5/6 | Normal breast epithelium co-expresses CK5/6; DCIS typically lacks it. | Same quote as row 2 (PMID 12474219). | |
| 4 | Benign lesion vs invasive carcinoma | Stain: p63 with smooth muscle myosin heavy chain or calponin | Benign mimics such as sclerosing lesions keep a myoepithelial layer around their glands; invasive carcinoma lacks it. | Lee et al., J Clin Pathol 2013 (PMID 23486609): "Myoepithelial markers, such as smooth muscle actin, smooth muscle myosin heavy chain and p63, are useful for distinguishing invasive carcinoma from sclerosing lesions and ductal carcinoma in situ (DCIS), and in the classification of papillary lesions." Werling et al., Am J Surg Pathol 2003 (PMID 12502930): "Identification of myoepithelial cells using antibodies to cytoskeletal proteins, such as smooth muscle myosin heavy chain (SMM-HC) and calponin, can play an important role in distinguishing invasive carcinoma from its histologic mimics." | |
| 5 | Normal tissue vs invasive carcinoma | Stain: p63 with smooth muscle myosin heavy chain or calponin | Normal ducts and lobules have a myoepithelial (basal) layer; invasive carcinoma lacks it. | Alkhayyat et al., Histopathology 2024 (PMID 38012539): "p63 protein is normally expressed in the nuclei of the basal cell layer of glandular organs, including breast, in squamous epithelium and in urothelium." Bonacho et al., Biotech Histochem 2020 (PMID 31502889): "Myoepithelial cell (MEC) stains are used to evaluate in situ and invasive carcinoma; calponin, smooth muscle myosin heavy chain and p63 are sensitive/specific markers." | |
| 6 | Normal tissue vs benign lesion | Correlate with imaging (no stain) | Check that the tissue matches the imaging target. If imaging showed a lesion but the tissue looks normal, the result may be discordant; a discordant suspicious case needs repeat biopsy or excision. | Wells et al., Radiographics 2023 (PMID 37676826): "Assessment of radiologic-pathologic concordance is vital in guiding appropriate management, as it enables identification of discordant results, minimizing the chance of misdiagnosis." Speer et al., J Breast Imaging 2023 (PMID 36778653): "Surgical excision or repeat core-needle biopsy is necessary for discordant suspicious cases." | |

Also useful to know:
- Are the two general steps the right ones, in the right order? Anything missing?
- Would you want the tool to say *when not* to order a stain (for example, when H&E alone is usually enough)?

The tool is a research demo for a hackathon and is not used for patient care.
