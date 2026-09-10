# Dataset governance

Code licensing and dataset licensing are separate. The repository's MIT license covers the
software only. It does not grant rights to source texts, a combined corpus, or a generated
database.

The currently linked Hugging Face files do not yet have a completed, repository-verifiable
source manifest. Their provenance and right-to-redistribute status must therefore be treated
as **unverified**. This statement is not a claim that any source is infringing; it records
that the evidence needed to make a positive licensing claim is missing.

Before a dataset is made or kept public:

1. Inventory every input source with a stable URL or archival identifier.
2. Record the exact license/terms evidence and the permitted transformations for each source.
3. Exclude sources whose use or redistribution right cannot be demonstrated.
4. Record SHA-256 hashes for inputs, the build command, code commit, output, and row counts.
5. Review whether generated databases or samples reproduce meaningful source passages.
6. Publish the completed `data/provenance.template.json` as `provenance.json` beside the data.
7. Add matching provenance, limitations, intended-use, and attribution sections to the
   Hugging Face dataset card.

If this review cannot be completed promptly, make the dataset private until it can.
