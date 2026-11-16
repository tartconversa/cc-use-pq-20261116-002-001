import csv
import io
import json
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP

from .errors import DomainError, require
from .storage import Store
from .validation import payload_digest


# A judge needs at least this many scored teams before a mean deviation is
# treated as a usable observation rather than a single anecdote.
MIN_TEAMS_PER_JUDGE = 2
# A paired judge-to-judge comparison needs this many teams in common.
MIN_COMMON_TEAMS = 2
# Inter-judge differences cannot be estimated from a one-person panel.
MIN_JUDGES_FOR_PANEL = 2

LEVELS = {'sufficient': 0, 'limited': 1, 'insufficient': 2}

RULES = {
    'normalization': (
        'Official totals are already out of 100. Each criterion is divided by '
        'its rubric maximum and shown as a percentage, so totals, criteria and '
        'groups share one scale.'),
    'peer_baseline': (
        'A judge deviation is the signed difference between that judge and the '
        'mean of the other active judges on the same team, averaged over the '
        'teams both scored. Positive means the judge scored higher.'),
    'min_teams_per_judge': MIN_TEAMS_PER_JUDGE,
    'min_common_teams': MIN_COMMON_TEAMS,
    'min_judges_for_panel': MIN_JUDGES_FOR_PANEL,
    'confidence': {
        'sufficient': (
            'At least two active judges, every group team scored, enough '
            'common teams, no revised cards and no upheld disputes.'),
        'limited': (
            'A paired comparison exists, but revised cards or upheld disputes '
            'are recorded for this judge.'),
        'insufficient': (
            'Panel size, coverage or shared sample is too small, or an open '
            'dispute remains. Treat the figures as descriptive, not as evidence '
            'of bias.'),
    },
    'official_standings': (
        'Official score, median and rank are copied from the published '
        'standings and are never recalculated by this analysis.'),
}

REASON_TEXT = {
    'inactive_judge': (
        'critical',
        'Judge is inactive but has a card in this group.',
        'Inactive cards do not form part of official standings; the diagnosis is provisional.'),
    'partial_coverage': (
        'limitation',
        'Judge scored {scored} of {teams} teams in the group.',
        'Paired deviations use only shared teams and do not represent the whole group; '
        'judge-level confidence is insufficient.'),
    'insufficient_teams': (
        'limitation',
        'Judge scored {scored} team(s), below the minimum of {minimum_teams}.',
        'A mean deviation from too few teams is not evidence of systematic bias.'),
    'single_judge_panel': (
        'limitation',
        'The group has fewer than {minimum_judges} active judges.',
        'No inter-judge difference can be estimated; scores are descriptive only.'),
    'no_common_teams': (
        'limitation',
        'Judge shares no team with another active judge.',
        'No paired comparison is possible.'),
    'insufficient_common_teams': (
        'limitation',
        'Judge shares only {common} team(s) with peers, below the minimum of '
        '{minimum_common}.',
        'The paired deviation is not stable enough to support a conclusion.'),
    'card_revised': (
        'caution',
        'Judge has {revised} revised card(s); the latest revision is used.',
        'Earlier revisions are excluded from the calculation and confidence is limited.'),
    'dispute_upheld': (
        'caution',
        '{upheld} upheld dispute(s) concern this judge; the published card was not rewritten.',
        'The concern limits confidence and must be read alongside the deviation, '
        'not as proof of bias.'),
    'open_dispute': (
        'critical',
        '{open} dispute(s) remain open.',
        'The diagnosis is provisional; open disputes normally block publication.'),
}


def _display(value: Decimal, places: str = '0.01') -> str:
    return str(Decimal(value).quantize(Decimal(places), rounding=ROUND_HALF_UP))


def _decimal(value) -> Decimal:
    return Decimal(str(value))


def _mean(values: list[Decimal]) -> Decimal:
    return sum(values, Decimal('0')) / Decimal(len(values))


def _stats(values: list[Decimal]) -> dict | None:
    if not values:
        return None
    mean = _mean(values)
    variance = sum(((value - mean) ** 2 for value in values), Decimal('0')) / Decimal(len(values))
    return {'count': len(values), 'mean': _display(mean),
            'min': _display(min(values)), 'max': _display(max(values)),
            'range': _display(max(values) - min(values)),
            'stddev': _display(variance.sqrt())}


class Consistency:
    """Frozen, advisory scoring-consistency analysis.

    The report is built once, inside the publication transaction, and stored in
    the immutable snapshot payload. Later backfills or re-publications create a
    new version; they cannot change an older version's report. Official
    standings are copied verbatim and are never recalculated here.
    """

    def __init__(self, store: Store):
        self.store = store

    # -- build at publication time -------------------------------------
    def build(self, db, event_id: str, scores: dict) -> dict:
        event = self.store.one(db, 'SELECT * FROM events WHERE id=?', (event_id,))
        groups = self.store.many(db, 'SELECT * FROM groups WHERE event_id=? ORDER BY code',
                                 (event_id,))
        teams = self.store.many(db, 'SELECT * FROM teams WHERE event_id=? ORDER BY id',
                                (event_id,))
        judges = self.store.many(db, 'SELECT * FROM judges WHERE event_id=? ORDER BY id',
                                 (event_id,))
        cards = self.store.many(db, 'SELECT * FROM score_cards WHERE event_id=? '
                                'ORDER BY team_id,judge_id', (event_id,))
        revisions = self.store.many(
            db, 'SELECT team_id,judge_id,COUNT(*) AS revision_count FROM score_history '
                'WHERE event_id=? GROUP BY team_id,judge_id', (event_id,))
        disputes = self.store.many(db, 'SELECT * FROM score_disputes WHERE event_id=? '
                                   'ORDER BY opened_at,id', (event_id,))

        official_groups = {group['code']: group for group in scores['groups']}
        judge_names = {judge['id']: judge['display_name'] for judge in judges}
        judge_active = {judge['id']: bool(judge['active']) for judge in judges}
        revision_info = {(row['team_id'], row['judge_id']): row for row in revisions}
        cards_by_team = defaultdict(list)
        for card in cards:
            cards_by_team[card['team_id']].append(card)

        rendered_groups = []
        for group in groups:
            code = group['code']
            maxima = {key: _decimal(value)
                      for key, value in json.loads(group['rubric']).items()}
            group_teams = [team for team in teams if team['group_code'] == code]
            group_team_ids = {team['id'] for team in group_teams}
            group_judges = [judge for judge in judges if judge['group_code'] == code]
            active_ids = sorted(judge['id'] for judge in group_judges if judge['active'])
            group_cards = [card for team in group_teams
                           for card in cards_by_team[team['id']]]
            cards_by_judge = defaultdict(list)
            for card in group_cards:
                cards_by_judge[card['judge_id']].append(card)
            panel_ids = sorted(set(active_ids) | set(cards_by_judge))
            official = official_groups.get(code, {'teams': [], 'eligible': False})
            official_teams = {team['team_id']: team for team in official['teams']}
            card_lookup = {(card['team_id'], card['judge_id']): card
                           for card in group_cards}
            norm_cache = {(card['team_id'], card['judge_id']):
                          self._normalized_criteria(card, maxima)
                          for card in group_cards}

            team_rows = []
            for team in group_teams:
                team_cards = sorted(cards_by_team[team['id']],
                                    key=lambda card: card['judge_id'])
                active_totals = [_decimal(card['total']) for card in team_cards
                                 if judge_active.get(card['judge_id'], False)]
                panel_mean = _mean(active_totals) if len(active_totals) >= 2 else None
                criterion_values = defaultdict(list)
                card_rows = []
                for card in team_cards:
                    normalized = norm_cache[(team['id'], card['judge_id'])]
                    revision = revision_info.get((team['id'], card['judge_id']), {})
                    raw_scores = json.loads(card['scores'])
                    card_rows.append({
                        'judge_id': card['judge_id'],
                        'judge_name': judge_names.get(card['judge_id'], card['judge_id']),
                        'active_judge': judge_active.get(card['judge_id'], False),
                        'revision': card['revision'],
                        'revision_count': revision.get('revision_count', 1),
                        'submitted_at': card['submitted_at'],
                        'total': _display(_decimal(card['total'])),
                        'criteria': {key: _display(_decimal(value))
                                     for key, value in sorted(raw_scores.items())},
                        'normalized_total': _display(_decimal(card['total'])),
                        'normalized_criteria': {key: _display(value)
                                                for key, value in sorted(normalized.items())},
                        'deviation_from_panel_mean': (
                            _display(_decimal(card['total']) - panel_mean)
                            if panel_mean is not None else None),
                    })
                    if not judge_active.get(card['judge_id'], False):
                        continue
                    for key, value in normalized.items():
                        criterion_values[key].append(value)
                official_row = official_teams.get(team['id'], {})
                team_rows.append({
                    'team_id': team['id'],
                    'team_name': team['name'],
                    'district': team['district'],
                    'official': {
                        'score': official_row.get('score'),
                        'median': official_row.get('median'),
                        'rank': official_row.get('rank'),
                        'card_count': official_row.get('card_count'),
                        'judge_count': official_row.get('judge_count'),
                        'complete': official_row.get('complete'),
                    },
                    'cards': card_rows,
                    'normalized_panel': _stats(active_totals),
                    'criteria_panel_means': {
                        key: _display(_mean(values))
                        for key, values in sorted(criterion_values.items()) if values},
                })

            judge_rows = []
            for judge_id in panel_ids:
                judge_cards = sorted(cards_by_judge[judge_id],
                                     key=lambda card: card['team_id'])
                scored = {card['team_id'] for card in judge_cards}
                deviations = []
                criterion_deviations = defaultdict(list)
                for card in judge_cards:
                    peer_cards = [
                        other for other in cards_by_team[card['team_id']]
                        if other['judge_id'] != judge_id
                        and judge_active.get(other['judge_id'], False)]
                    if not peer_cards:
                        continue
                    value = _decimal(card['total'])
                    deviations.append(value - _mean(
                        [_decimal(other['total']) for other in peer_cards]))
                    normalized = norm_cache[(card['team_id'], judge_id)]
                    for key, criterion_value in normalized.items():
                        peer_values = [
                            norm_cache[(card['team_id'], other['judge_id'])][key]
                            for other in peer_cards
                            if key in norm_cache[(card['team_id'], other['judge_id'])]]
                        if peer_values:
                            criterion_deviations[key].append(
                                criterion_value - _mean(peer_values))
                coverage = (Decimal(len(scored)) / Decimal(len(group_teams))
                            if group_teams else Decimal('0'))
                peer_comparison = None
                if deviations:
                    peer_comparison = {
                        'common_teams': len(deviations),
                        'mean_signed_deviation': _display(_mean(deviations)),
                        'mean_absolute_deviation': _display(
                            _mean([abs(value) for value in deviations])),
                        'max_positive_deviation': _display(max(deviations)),
                        'max_negative_deviation': _display(min(deviations)),
                        'criteria': {
                            key: {
                                'common_teams': len(values),
                                'mean_signed_deviation': _display(_mean(values)),
                            }
                            for key, values in sorted(criterion_deviations.items()) if values},
                        'interpretation': ('positive means this judge scored above the '
                                           'other active judges on the same teams'),
                    }
                revised_cards = sum(
                    1 for card in judge_cards
                    if revision_info.get((card['team_id'], judge_id), {})
                    .get('revision_count', 1) > 1)
                max_revision = max((card['revision'] for card in judge_cards), default=0)
                judge_disputes = [
                    dispute for dispute in disputes
                    if dispute['card_judge_id'] == judge_id
                    and dispute['team_id'] in group_team_ids]
                dispute_counts = Counter(dispute['state'] for dispute in judge_disputes)

                reasons = []
                level = 'sufficient'

                def downgrade(target: str) -> None:
                    nonlocal level
                    if LEVELS[target] > LEVELS[level]:
                        level = target

                if not judge_active.get(judge_id, False):
                    downgrade('insufficient')
                    reasons.append('inactive_judge')
                if len(scored) < len(group_teams):
                    downgrade('insufficient')
                    reasons.append('partial_coverage')
                if len(scored) < MIN_TEAMS_PER_JUDGE:
                    downgrade('insufficient')
                    reasons.append('insufficient_teams')
                if len(active_ids) < MIN_JUDGES_FOR_PANEL:
                    downgrade('insufficient')
                    reasons.append('single_judge_panel')
                if not deviations:
                    downgrade('insufficient')
                    reasons.append('no_common_teams')
                elif len(deviations) < MIN_COMMON_TEAMS:
                    downgrade('insufficient')
                    reasons.append('insufficient_common_teams')
                if revised_cards:
                    downgrade('limited')
                    reasons.append('card_revised')
                if dispute_counts.get('upheld', 0):
                    downgrade('limited')
                    reasons.append('dispute_upheld')
                if dispute_counts.get('open', 0):
                    downgrade('insufficient')
                    reasons.append('open_dispute')

                judge_rows.append({
                    'judge_id': judge_id,
                    'judge_name': judge_names.get(judge_id, judge_id),
                    'active': judge_active.get(judge_id, False),
                    'cards_submitted': len(judge_cards),
                    'teams_in_group': len(group_teams),
                    'coverage': _display(coverage),
                    'coverage_complete': len(scored) == len(group_teams),
                    'mean_normalized_total': (
                        _display(_mean([_decimal(card['total'])
                                        for card in judge_cards]))
                        if judge_cards else None),
                    'peer_comparison': peer_comparison,
                    'revisions': {'cards_revised': revised_cards,
                                  'max_revision': max_revision},
                    'disputes': {
                        'total': len(judge_disputes),
                        'open': dispute_counts.get('open', 0),
                        'upheld': dispute_counts.get('upheld', 0),
                        'rejected': dispute_counts.get('rejected', 0),
                        'items': [{
                            'id': dispute['id'],
                            'team_id': dispute['team_id'],
                            'state': dispute['state'],
                            'resolution': dispute.get('resolution'),
                            'card_revision': dispute['card_revision'],
                            'opened_at': dispute['opened_at'],
                            'closed_at': dispute['closed_at'],
                        } for dispute in judge_disputes],
                    },
                    'reliability': {'level': level, 'reasons': reasons},
                })

            pairwise = []
            for index, judge_a in enumerate(active_ids):
                for judge_b in active_ids[index + 1:]:
                    common = sorted(
                        team_id for team_id in group_team_ids
                        if (team_id, judge_a) in card_lookup
                        and (team_id, judge_b) in card_lookup)
                    differences = [
                        _decimal(card_lookup[(team_id, judge_a)]['total'])
                        - _decimal(card_lookup[(team_id, judge_b)]['total'])
                        for team_id in common]
                    pairwise.append({
                        'judge_a': judge_a,
                        'judge_b': judge_b,
                        'common_teams': len(common),
                        'mean_signed_difference': (
                            _display(_mean(differences)) if differences else None),
                        'mean_absolute_difference': (
                            _display(_mean([abs(value) for value in differences]))
                            if differences else None),
                        'comparable': len(common) >= MIN_COMMON_TEAMS,
                        'interpretation': ('positive means judge_a scored higher '
                                           'than judge_b'),
                    })

            sample = {
                'teams_in_group': len(group_teams),
                'active_judges': len(active_ids),
                'cards_present': len(group_cards),
                'cards_expected': len(group_teams) * len(active_ids),
                'cards_from_inactive_judges': sum(
                    1 for card in group_cards
                    if not judge_active.get(card['judge_id'], False)),
                'complete': (bool(group_teams) and bool(official.get('eligible'))
                             and all(team['complete'] for team in official['teams'])),
            }
            active_rows = [row for row in judge_rows if row['active']]
            if active_rows:
                panel_confidence = max(
                    active_rows,
                    key=lambda row: LEVELS[row['reliability']['level']]
                )['reliability']['level']
            else:
                panel_confidence = 'insufficient'
            if len(active_ids) < MIN_JUDGES_FOR_PANEL:
                panel_confidence = 'insufficient'

            diagnostics = []
            if len(active_ids) < MIN_JUDGES_FOR_PANEL:
                diagnostics.append(self._diagnostic(
                    'single_judge_panel', code, None, {
                        'minimum_judges': MIN_JUDGES_FOR_PANEL}))
            if not sample['complete']:
                diagnostics.append({
                    'code': 'incomplete_group',
                    'severity': 'critical',
                    'group': code,
                    'judge_id': None,
                    'message': ('Not every team has exactly one card from every '
                                'active judge.'),
                    'effect': ('Official standings omit incomplete teams; '
                               'consistency conclusions are provisional.'),
                })
            for row in judge_rows:
                context = {
                    'scored': row['cards_submitted'],
                    'teams': row['teams_in_group'],
                    'minimum_teams': MIN_TEAMS_PER_JUDGE,
                    'minimum_judges': MIN_JUDGES_FOR_PANEL,
                    'common': (row['peer_comparison']['common_teams']
                               if row['peer_comparison'] else 0),
                    'minimum_common': MIN_COMMON_TEAMS,
                    'revised': row['revisions']['cards_revised'],
                    'upheld': row['disputes']['upheld'],
                    'open': row['disputes']['open'],
                }
                for reason in row['reliability']['reasons']:
                    if reason == 'single_judge_panel':
                        continue
                    diagnostics.append(self._diagnostic(
                        reason, code, row['judge_id'], context))
            for pair in pairwise:
                if not pair['comparable']:
                    diagnostics.append({
                        'code': 'insufficient_pairwise_sample',
                        'severity': 'limitation',
                        'group': code,
                        'judge_id': None,
                        'message': (f"Judges {pair['judge_a']} and {pair['judge_b']} "
                                    f"share {pair['common_teams']} team(s), below "
                                    f'the minimum of {MIN_COMMON_TEAMS}.'),
                        'effect': ('Their difference is shown for context but is not '
                                   'a stable comparison.'),
                    })

            rendered_groups.append({
                'code': code,
                'label': group['label'],
                'rubric': {key: _display(value) for key, value in sorted(maxima.items())},
                'min_judges': group['min_judges'],
                'max_judges': group['max_judges'],
                'sample': sample,
                'panel_confidence': panel_confidence,
                'teams': team_rows,
                'judges': judge_rows,
                'pairwise': pairwise,
                'diagnostics': diagnostics,
            })

        report = {
            'event_id': event_id,
            'source_revision': event['revision'],
            'caliber': {
                'official_total_scale': '100',
                'normalized_scale': 'percent of the applicable maximum (0-100)',
                'total_comparison': ('judge totals are already out of 100, so totals '
                                     'are compared directly'),
                'criterion_comparison': ('each criterion is divided by its rubric '
                                         'maximum before comparison'),
                'peer_baseline': ('mean of the other active judges on the same team, '
                                  'paired by team'),
            },
            'rules': RULES,
            'official_standings_unchanged': True,
            'revision_effect': ('Card details use the final submitted revision in this '
                                'published version. Earlier revisions remain in score '
                                'history but do not enter these calculations.'),
            'dispute_effect': ('Resolved disputes are reported per judge. An upheld '
                               'dispute records a concern but does not rewrite the '
                               'published card; it limits confidence. Open disputes '
                               'would block publication and make the diagnosis '
                               'provisional.'),
            'groups': rendered_groups,
        }
        report['report_digest'] = payload_digest(report)
        return report

    @staticmethod
    def _normalized_criteria(card, maxima: dict[str, Decimal]) -> dict[str, Decimal]:
        raw = json.loads(card['scores'])
        return {key: _decimal(raw[key]) / maxima[key] * Decimal('100')
                for key in maxima if key in raw}

    @staticmethod
    def _diagnostic(reason: str, group_code: str, judge_id: str | None,
                    context: dict) -> dict:
        severity, message, effect = REASON_TEXT[reason]
        return {'code': reason, 'severity': severity, 'group': group_code,
                'judge_id': judge_id, 'message': message.format(**context),
                'effect': effect}

    # -- read a frozen report ------------------------------------------
    def report(self, event_id: str, version: int | None = None,
               group_code: str | None = None, judge_id: str | None = None) -> dict:
        with self.store.connection() as db:
            self.store.one(db, 'SELECT id FROM events WHERE id=?', (event_id,))
            if version is None:
                row = db.execute(
                    'SELECT payload FROM snapshots WHERE event_id=? AND status=? '
                    'ORDER BY version DESC LIMIT 1',
                    (event_id, 'published')).fetchone()
            else:
                require(type(version) is int and version > 0, 'invalid_version',
                        'Version must be positive')
                row = db.execute(
                    'SELECT payload FROM snapshots WHERE event_id=? AND version=? '
                    'AND status=?', (event_id, version, 'published')).fetchone()
        if not row:
            raise DomainError('snapshot_not_found', 'Published snapshot does not exist', 404)
        payload = json.loads(row['payload'])
        report = payload.get('consistency')
        if not report:
            raise DomainError(
                'analysis_unavailable',
                'This published version predates consistency analysis; '
                'publish a new version to record it', 409)
        stored_digest = report.get('report_digest')
        digest_valid = stored_digest == payload_digest(
            {key: value for key, value in report.items() if key != 'report_digest'})

        groups = report['groups']
        if group_code is not None:
            groups = [group for group in groups if group['code'] == group_code]
            require(bool(groups), 'not_found',
                    'Group is not present in this published version', 404)
        if judge_id is not None:
            selected = []
            for group in groups:
                if not any(row['judge_id'] == judge_id for row in group['judges']):
                    continue
                selected.append({
                    **group,
                    'judges': [row for row in group['judges']
                               if row['judge_id'] == judge_id],
                    'pairwise': [pair for pair in group['pairwise']
                                 if judge_id in (pair['judge_a'], pair['judge_b'])],
                    'diagnostics': [item for item in group['diagnostics']
                                    if item.get('judge_id') in (None, judge_id)],
                })
            require(bool(selected), 'not_found',
                    'Judge is not present in this published version', 404)
            groups = selected

        return {**report,
                'version': payload.get('version'),
                'published_at': payload.get('published_at'),
                'digest_valid': digest_valid,
                'frozen': True,
                'filters': {'group': group_code, 'judge': judge_id},
                'groups': groups}

    def csv(self, report: dict) -> str:
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(['event_id', 'version', 'group', 'team_id', 'team_name',
                         'official_rank', 'official_score', 'official_median',
                         'judge_id', 'judge_name', 'judge_active', 'revision',
                         'revision_count', 'total', 'normalized_total',
                         'deviation_from_panel_mean'])
        for group in report['groups']:
            for team in group['teams']:
                official = team['official']
                if not team['cards']:
                    writer.writerow([report['event_id'], report['version'], group['code'],
                                     team['team_id'], team['team_name'],
                                     official['rank'], official['score'],
                                     official['median'], '', '', '', '', '', '', '', ''])
                    continue
                for card in team['cards']:
                    writer.writerow([report['event_id'], report['version'], group['code'],
                                     team['team_id'], team['team_name'],
                                     official['rank'], official['score'],
                                     official['median'], card['judge_id'],
                                     card['judge_name'], card['active_judge'],
                                     card['revision'], card['revision_count'],
                                     card['total'], card['normalized_total'],
                                     card['deviation_from_panel_mean']])
        return stream.getvalue()
