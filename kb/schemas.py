"""Pydantic schemas for the knowledge base (Part 1)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

RefKind = Literal["listed", "inline"]
ResolveStatus = Literal["unresolved", "resolved", "ambiguous", "not_found"]
FetchStatus = Literal["pending", "downloaded", "missing", "paywalled", "failed"]


class KosmosClaim(BaseModel):
    """A claim from the Kosmos report, with the citation markers that support it."""

    claim_id: str
    report_id: str
    query: str  # L1 | L2 | L3 | meta
    section: str
    text: str
    markers: list[str] = Field(default_factory=list)  # ["3.7", "3.8"]
    marker_groups: list[str] = Field(default_factory=list)  # ["3"]
    inline_mentions: list[str] = Field(default_factory=list)  # ["Baarslag et al. 2016"]


class Reference(BaseModel):
    """A work cited by Kosmos.

    `listed`: has a formal entry in the report's References section.
    `inline`: only named in the body (author + year); resolved by search.
    """

    ref_id: str
    report_id: str
    ref_kind: RefKind
    marker_group: Optional[str] = None
    markers: list[str] = Field(default_factory=list)
    raw_entry: Optional[str] = None
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    mention_string: Optional[str] = None  # "Baarslag et al. 2016"
    context_quotes: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    n_citations: int = 0
    resolve_status: ResolveStatus = "unresolved"
    needs_review: bool = False
    review_note: Optional[str] = None
    # filled by kb.resolve_metadata
    openalex_id: Optional[str] = None
    resolved_title: Optional[str] = None
    resolved_authors: list[str] = Field(default_factory=list)
    resolved_year: Optional[int] = None
    resolved_venue: Optional[str] = None
    is_oa: Optional[bool] = None
    oa_pdf_url: Optional[str] = None
    alt_pdf_urls: list[str] = Field(default_factory=list)
    match_score: Optional[float] = None
    # filled by kb.fetch_papers
    paper_id: Optional[str] = None
    fetch_status: FetchStatus = "pending"
    pdf_path: Optional[str] = None
    pdf_sha256: Optional[str] = None
    fetch_note: Optional[str] = None
