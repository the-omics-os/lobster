# Commercial Licensing FAQ

## What is AGPL-3.0-or-later?

The GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later) is a free, copyleft open-source license that ensures all users have the freedom to use, study, share, and modify the software. The AGPL extends the GPL by including a network copyleft provision: if you modify Lobster AI and run it as a network service, you must make the modified source code available to users of that service.

## Why did Lobster AI change from Apache-2.0 to AGPL-3.0-or-later?

Lobster AI depends on GPL-licensed libraries (specifically `igraph` and `leidenalg`) which are essential for bioinformatics graph analysis and clustering. The Apache-2.0 license was incompatible with these GPL dependencies. By adopting AGPL-3.0-or-later, we:

1. **Ensure legal compliance** with our GPL dependencies
2. **Maintain open-source principles** - AGPL is an OSI-approved license
3. **Protect community contributions** - modifications to network-deployed code must be shared
4. **Support our business model** - clearer separation between open-source and commercial offerings

## Can I still use Lobster AI for free?

**Yes!** You can use Lobster AI under the AGPL-3.0-or-later license at no cost. The license grants you:

- Freedom to run the software for any purpose
- Freedom to study and modify the source code
- Freedom to distribute copies
- Freedom to distribute modified versions

## When do I need a commercial license?

You may want a commercial license if:

1. **You modify Lobster AI and run it as a service** but prefer not to share your modifications under AGPL terms
2. **Your organization's legal policies** prohibit AGPL software
3. **You need dedicated support** - priority bug fixes, feature requests, or consulting
4. **You want additional warranties or indemnification** beyond what the AGPL provides
5. **You're integrating Lobster AI into proprietary software** and cannot comply with AGPL's copyleft requirements

## What does the commercial license provide?

Our commercial license offers:

- **Reduced copyleft obligations** - use Lobster AI in proprietary systems without sharing modifications
- **Dedicated support** - priority access to engineering team, faster bug fixes, and feature development
- **Service Level Agreements (SLAs)** - guaranteed response times and uptime commitments
- **Enhanced warranties** - stronger guarantees than the "AS IS" AGPL terms
- **Custom integrations** - tailored features for your organization's specific needs
- **Training and onboarding** - workshops and documentation for your team

## What are the AGPL-3.0-or-later compliance requirements?

If you use or modify Lobster AI, you must:

1. **Preserve all copyright and license notices** in the code
2. **Provide a copy of the AGPL-3.0-or-later license** with any distribution
3. **Make source code available** if you distribute the software
4. **If you modify and deploy as a network service**: Offer your modified source code to users of that service (this is the key AGPL addition to GPL)

## Can I use Lobster AI in my research paper/thesis?

**Yes!** Academic research use is fully supported under AGPL-3.0-or-later. You can:

- Use Lobster AI for your research analysis
- Include results and figures in publications
- Cite Lobster AI in your papers
- Modify the code for your research needs

You only need to share modifications if you deploy a modified version as a network service that others access.

## Can my company use Lobster AI internally?

**Yes!** Internal use within your organization (running on your own servers for your own employees) is permitted under AGPL-3.0-or-later. You do not need to share modifications unless you provide the software as a service to users outside your organization.

## How does AGPL affect my own analysis results?

**Your data and analysis results remain yours.** The AGPL applies to the Lobster AI software itself, not to:

- Your input data
- Your analysis results
- Figures and visualizations you create
- Research papers or reports you write

You own your scientific outputs completely.

## How is Lobster AI's business model sustainable with AGPL?

Our dual-licensing model is proven by successful companies like MongoDB, Elastic, and MariaDB:

1. **Open-source community edition** (AGPL-3.0-or-later) drives adoption, testing, and contributions
2. **Commercial licenses** provide enterprise features, support, and revenue
3. **Consulting and services** for custom bioinformatics platform implementations

This model allows us to:
- Maintain a vibrant open-source community
- Provide enterprise-grade support for organizations
- Invest in continued development and research

## What if I'm already using Lobster AI under Apache-2.0?

Lobster AI version 0.3.0 and later are licensed under AGPL-3.0-or-later. If you are using version 0.2.x or earlier, those versions were released under Apache-2.0 and you may continue using them under those terms. However:

- Versions prior to 0.3.0 have a license compatibility issue with GPL dependencies
- We recommend upgrading to 0.3.0+ for legal compliance
- Future updates, bug fixes, and features will only be available in AGPL-licensed versions

## How do I get a commercial license?

Contact us to discuss your organization's needs:

**Email**: info@omics-os.com
**Website**: https://www.omics-os.com

We offer flexible commercial licensing tailored to:
- Startups and biotech companies ($6K-$18K/year typical range)
- Research institutions (special academic pricing available)
- Enterprise biopharma ($18K-$30K/year with SLAs and compliance)

## Additional Resources

- **Full AGPL-3.0 License Text**: See the `LICENSE` file in the repository or visit https://www.gnu.org/licenses/agpl-3.0
- **AGPL Compliance Guide**: https://www.gnu.org/licenses/gpl-faq.html
- **Lobster AI Documentation**: https://github.com/the-omics-os/lobster.wiki
- **Commercial Licensing Inquiries**: info@omics-os.com

---

*Last updated: 2025-11-18*
