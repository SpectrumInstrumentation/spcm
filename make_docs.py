from pathlib import Path
import pdoc
import os

import spcm

if __name__ == '__main__':
    os.environ["PDOC_ALLOW_EXEC"] = "1"

    doc_path = Path("docs")

    doc = pdoc.doc.Module(spcm)

    pdoc.render.configure(favicon="https://spectrum-instrumentation.com/img/favicon.ico",
                          logo="https://spectrum-instrumentation.com/img/logo-complete.png",
                          logo_link="https://spectrum-instrumentation.com/",
                          mermaid=True,
                          docformat="numpy",
                          footer_text="Spectrum Instrumentation GmbH",
                          search=False,
                          show_source=True,)
    
    html = pdoc.render.html_module(module=doc, all_modules={"spcm": doc})
    html_index = pdoc.render.html_index(all_modules={"spcm": doc})

    with open(doc_path / "spcm.html", "w") as f:
        f.write(html)
    with open(doc_path / "index.html", "w") as f:
        f.write(html_index)


