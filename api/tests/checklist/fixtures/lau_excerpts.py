"""A curated stand-in for the ingested LAU corpus, block by block.

Each excerpt is a faithful fragment of the article's current consolidated
redaction and contains, verbatim, the citation the checklist quotes from that
block. It lets the validator run fully network-free while still exercising a real
substring match: a typo in a shipped citation would fail against these texts. The
operational guarantee against the *live* corpus is the ``validate-checklist`` CLI.
"""

LAU_NORM_ID = "BOE-A-1994-26003"

LAU_EXCERPTS: dict[str, str] = {
    "a6": (
        "Son nulas, y se tendrán por no puestas, las estipulaciones que modifiquen en "
        "perjuicio del arrendatario o subarrendatario las normas del presente Título, salvo "
        "los casos en que la propia norma expresamente lo autorice."
    ),
    "a8": (
        "La vivienda arrendada solo se podrá subarrendar de forma parcial y previo "
        "consentimiento escrito del arrendador. El precio del subarriendo no podrá exceder, "
        "en ningún caso, del que corresponda al arrendamiento."
    ),
    "a9": (
        "La duración del arrendamiento será libremente pactada por las partes. Si esta fuera "
        "inferior a cinco años, o inferior a siete años si el arrendador fuese persona "
        "jurídica, llegado el día del vencimiento del contrato, este se prorrogará "
        "obligatoriamente por plazos anuales hasta que el arrendamiento alcance una duración "
        "mínima de cinco años, o de siete años si el arrendador fuese persona jurídica, salvo "
        "que el arrendatario manifieste su voluntad de no renovarlo."
    ),
    "a10": (
        "Si llegada la fecha de vencimiento del contrato, o de cualquiera de sus prórrogas, "
        "una vez transcurridos como mínimo cinco años de duración de aquel, o siete años si el "
        "arrendador fuese persona jurídica, ninguna de las partes hubiese notificado a la otra "
        "su voluntad de no renovarlo, el contrato se prorrogará obligatoriamente por plazos "
        "anuales hasta un máximo de tres años más."
    ),
    "a11": (
        "El arrendatario podrá desistir del contrato de arrendamiento, una vez que hayan "
        "transcurrido al menos seis meses, siempre que se lo comunique al arrendador con una "
        "antelación mínima de treinta días."
    ),
    "a12": (
        "Si el arrendatario manifestara su voluntad de no renovar el contrato o de desistir de "
        "él, sin el consentimiento del cónyuge que conviviera con dicho arrendatario, el "
        "arrendamiento podrá continuar en beneficio del cónyuge que conviva con aquel."
    ),
    "a14": (
        "El adquirente de una vivienda arrendada quedará subrogado en los derechos y "
        "obligaciones del arrendador durante los cinco primeros años de vigencia del contrato, "
        "o siete años si el arrendador anterior fuese persona jurídica."
    ),
    "a15": (
        "En los casos de nulidad del matrimonio, separación judicial o divorcio del "
        "arrendatario, el cónyuge no arrendatario podrá continuar en el uso de la vivienda "
        "arrendada cuando le sea atribuida de acuerdo con lo dispuesto en la legislación civil."
    ),
    "a16": (
        "En caso de muerte del arrendatario, podrán subrogarse en el contrato las personas que "
        "señala este artículo. El arrendamiento se extinguirá si en el plazo de tres meses "
        "desde la muerte del arrendatario el arrendador no recibe notificación por escrito del "
        "hecho del fallecimiento, con certificado registral de defunción, y de la identidad "
        "del subrogado."
    ),
    "a17": (
        "La renta será la que libremente estipulen las partes. El arrendador queda obligado a "
        "entregar recibo del pago, salvo que se hubiera pactado que éste se realice mediante "
        "procedimientos que acrediten el efectivo cumplimiento de la obligación de pago por el "
        "arrendatario."
    ),
    "a18": (
        "Durante la vigencia del contrato, la renta solo podrá ser actualizada por el "
        "arrendador o el arrendatario en la fecha en que se cumpla cada año de vigencia del "
        "contrato, en los términos pactados por las partes. En defecto de pacto expreso, no se "
        "aplicará actualización de rentas a los contratos."
    ),
    "a19": (
        "La realización por el arrendador de obras de mejora, transcurrido el plazo mínimo, le "
        "dará derecho, salvo pacto en contrario, a elevar la renta anual, sin que pueda exceder "
        "el aumento de la renta del veinte por ciento de la renta vigente en aquel momento."
    ),
    "a20": (
        "Los gastos de gestión inmobiliaria y los de formalización del contrato serán a cargo "
        "del arrendador. El pago de los gastos generales para el adecuado sostenimiento del "
        "inmueble podrá ser asumido por el arrendatario cuando las partes así lo acuerden; para "
        "su validez, este pacto deberá constar por escrito y determinar el importe anual de "
        "dichos gastos a la fecha del contrato."
    ),
    "a21": (
        "El arrendador está obligado a realizar, sin derecho a elevar por ello la renta, todas "
        "las reparaciones que sean necesarias para conservar la vivienda en las condiciones de "
        "habitabilidad para servir al uso convenido."
    ),
    "a22": (
        "El arrendatario estará obligado a soportar la realización por el arrendador de obras "
        "de mejora cuya ejecución no pueda razonablemente diferirse hasta la conclusión del "
        "arrendamiento. El arrendador que se proponga realizar tales obras deberá notificar por "
        "escrito al arrendatario, al menos con tres meses de antelación, su naturaleza, "
        "comienzo, duración y coste previsible."
    ),
    "a24": (
        "El arrendatario, previa notificación escrita al arrendador, podrá realizar en el "
        "interior de la vivienda aquellas obras o actuaciones necesarias para que pueda ser "
        "utilizada de forma adecuada y acorde a la discapacidad o a la edad superior a setenta "
        "años, tanto del propio arrendatario como de su cónyuge o de quienes con él convivan."
    ),
    "a25": (
        "En caso de venta de la vivienda arrendada, tendrá el arrendatario derecho de "
        "adquisición preferente sobre la misma, en las condiciones previstas en los apartados "
        "siguientes."
    ),
    "a26": (
        "Cuando la ejecución en la vivienda arrendada de obras de conservación o de obras "
        "acordadas por una autoridad competente la hagan inhabitable, tendrá el arrendatario "
        "la opción de suspender el contrato o de desistir del mismo, sin indemnización alguna."
    ),
    "a36": (
        "A la celebración del contrato será obligatoria la exigencia y prestación de fianza en "
        "metálico en cantidad equivalente a una mensualidad de renta en el arrendamiento de "
        "viviendas y de dos en el arrendamiento para uso distinto del de vivienda."
    ),
}
