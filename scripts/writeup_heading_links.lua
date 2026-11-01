-- Makes each write-up heading a link to itself, so a section's URL can be copied by clicking it. Used by build_writeup.sh.
local css = [[
<style>
a.heading-link { color: inherit; text-decoration: none; }
a.heading-link:hover::after { content: " #"; opacity: 0.4; }
</style>]]

function Header(header)
  header.content = { pandoc.Link(header.content, "#" .. header.identifier, "", { class = "heading-link" }) }
  return header
end

function Meta(meta)
  meta["header-includes"] = pandoc.MetaBlocks({ pandoc.RawBlock("html", css) })
  return meta
end
