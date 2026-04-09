# GRACE @ IberLEF 2026 - Rules snapshot

**Fetched:** 2026-04-09T16:59:59.006331+00:00

Source of truth for whether closed-API LLMs can be used on the final
test-set submission. Re-run ``scripts/verify_rules.py`` when organizers
update the Codabench Terms tab.

---

## Codabench competition (HTML)
**URL:** https://www.codabench.org/competitions/13280/

**Matched snippets:**
```
<a href="https://github.com/codalab/codalab-competitions/wiki/Privacy" class="item">Privacy and Terms</a>
<a href="/api/docs/" class="item">API Docs</a>
API: "/api/",
ANALYTICS_API: function (query_parameters) {
url: `${URLS.API}competitions/?search={query}`
```

## Codabench API (JSON endpoint fallback)
**URL:** https://www.codabench.org/api/competitions/13280/

**Matched snippets:**
```
{"id":13280,"title":"GRACE @ IberLEF 2026: Granular Recognition of Argumentative Clinical Evidence","published":true,"created_by":"idelaiglesia004","owner_display_name":"Iker","created_when":"2026-01-28T08:04:24.869711Z","logo":"https://miniodis-rproxy.lisn.upsaclay.fr/coda-v2-prod-public/logos/2026-02-10-1770718676/0778111a4fee/GRACE_Version_G2.png?AWSAccessKeyId=codabench-prod&Signature=RmCkiJHS9pIk%2Fq%2BVI1W0Nkoy77Q%3D&Expires=1775757602","logo_icon":"https://miniodis-rproxy.lisn.upsaclay.fr
```

## Corpora-list announcement
**URL:** http://www.mail-archive.com/corpora@list.elra.info/msg05615.html

**No rules-related snippets found automatically.**

## IberLEF 2026 tasks index
**URL:** https://sites.google.com/view/iberlef-2026/tasks

**Matched snippets:**
```
"><meta itemprop="url" content="https://sites.google.com/view/iberlef-2026/tasks"><meta itemprop="thumbnailUrl" content="https://lh3.googleusercontent.com/sitesv/APaQ0SQhOD8uQmyI9gbhrw5PVvCfoTb3TSphAMHzUJPvH3jK6oLq81poG7qnhkEOJ7qFG4co-Jb5fheif9mdbgdTV0tYim1KM06meQ23ujm5S7Khh6GuMbYvUlTusJbGYpfoEzcJWodTlyf0eKeEdvzyk_f0vLYE1wqlePwdhrO6PeLodP_T3pA9d5RZrcQ=w16383"><meta itemprop="image" content="https://lh3.googleusercontent.com/sitesv/APaQ0SQhOD8uQmyI9gbhrw5PVvCfoTb3TSphAMHzUJPvH3jK6oLq81poG7qnhkEOJ
a[0].toLowerCase();if(b.indexOf("on")===0||"on".indexOf(b)===0)throw Error("Prefix '"+a[0]+"' does not guarantee the attribute to be safe as it is also a prefix for event handler attributesPlease use 'addEventListener' to set event handlers.");ka.forEach(function(c){if(c.indexOf(b)===0)throw Error("Prefix '"+a[0]+"' does not guarantee the attribute to be safe as it is also a prefix for the security sensitive attribute '"+(c+"'. Please use native or safe DOM APIs to set the attribute."));});retur
function yb(a){function b(f,g){f&c&&d.push(g)}var c=Ea(a,"state is only maintained on arrays.")[wb]|0,d=[];b(1,"IS_REPEATED_FIELD");b(2,"IS_IMMUTABLE_ARRAY");b(4,"IS_API_FORMATTED");b(512,"STRING_FORMATTED");b(1024,"GBIGINT_FORMATTED");b(1024,"BINARY");b(8,"ONLY_MUTABLE_VALUES");b(16,"UNFROZEN_SHARED");b(32,"MUTABLE_REFERENCES_ARE_OWNED");b(64,"CONSTRUCTED");b(128,"HAS_MESSAGE_ID");b(256,"FROZEN_ARRAY");b(2048,"HAS_WRAPPER");b(4096,"MUTABLE_SUBSTRUCTURES");b(8192,"KNOWN_MAP_ARRAY");c&64&&(F(c&64
a=c>>14&1023||536870912,a!==536870912&&d.push("pivot: "+a));return d.join(",")};var zb=typeof Symbol!="undefined"&&typeof Symbol.hasInstance!="undefined";Object.freeze({});var Ab=function(){throw Error("please construct maps as mutable then call toImmutable");};if(zb){var Bb=function(){throw Error("Cannot perform instanceof checks on ImmutableMap: please use isImmutableMap or isMutableMap to assert on the mutability of a map. See go/jspb-api-gotchas#immutable-classes for more information");},Cb=
!1},Ub=function(a){return a.startsWith("https://uberproxy-pen-redirect.corp.google.com/uberproxy/pen?url=")?a.substr(65):a},Na={Ja:"k",fa:"ck",Ea:"m",oa:"exm",ma:"excm",Z:"am",aa:"amc",Ca:"mm",Ia:"rt",va:"d",na:"ed",Oa:"sv",ga:"deob",ba:"cb",da:"ccb",ca:"cbi",Ma:"rs",Ka:"sdch",wa:"im",ha:"dg",ka:"br",ja:"br-d",la:"rb",Ua:"zs",Ta:"wt",pa:"ee",Na:"sm",Da:"md",sa:"gssmodulesetproto",Sa:"ujg",Ra:"sp",La:"slk",ia:"dti",xa:"ic"};function Xb(){var a=x._F_jsUrl?"":"base-js";a=a===void 0?"":a;var b="";va
```

---

## Manual classification (fill in after reading above)

- [ ] Closed APIs (Claude, GPT-4) allowed on final test-set submission: **YES / NO**
- [ ] Extra pretrained open-weights models allowed: **YES / NO**
- [ ] Cross-lingual data augmentation allowed: **YES / NO**
- [ ] Ensembling across multiple models allowed: **YES / NO**
- [ ] Daily submission cap: **N**
- [ ] System paper required: **YES / NO**

Commit this file after filling in. If closed APIs are forbidden,
the Track 2 distilled student becomes the primary final-submission path.
