import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from community_fitness.api import create_app
from community_fitness.consistency import Consistency
from community_fitness.errors import DomainError
from tests.helpers import Scenario, START, END


class ConsistencyTests(Scenario):
    def setUp(self):
        super().setUp()
        self.consistency = Consistency(self.store)

    def seed_biased(self, *, judges=3, teams=4):
        """Judge j0 systematically scores about 8 points above the panel."""
        self.competition.create_event('evt', 'Showcase', START, END, 'Asia/Shanghai')
        self.competition.add_group('evt', 'dance', 'Dance', {'form': 60, 'rhythm': 40},
                                   2, 5)
        for number in range(teams):
            self.competition.register_team('evt', f't{number}', 'dance', f'Team {number}',
                                           'Qingpu', 3)
        for number in range(judges):
            self.competition.assign_judge('evt', f'j{number}', f'Judge {number}', 'dance')
        self.services.add_zone('evt', 'tea', 'Tea counter', 'tea',
                               '2026-10-26T09:00:00+08:00', '2026-10-26T18:00:00+08:00')
        self.competition.change_state('evt', 'open', self.revision())
        self.sessions.open('evt', 'dance', self.revision())
        for team in range(teams):
            for judge in range(judges):
                offset = 6 if judge == 0 else 0
                self.competition.submit_score(
                    'evt', f't{team}', f'j{judge}',
                    {'form': 44 + offset, 'rhythm': 30 + offset // 2},
                    f's-{team}-{judge}')

    def publish_after_scores(self):
        self.close()
        return self.publish()

    # -- fairness / same caliber ---------------------------------------
    def test_report_never_changes_official_standings(self):
        self.seed_biased()
        snapshot = self.publish_after_scores()
        report = snapshot['consistency']
        self.assertTrue(report['official_standings_unchanged'])
        self.assertEqual(report['source_revision'], snapshot['event']['revision'])
        for group in report['groups']:
            official = next(item for item in snapshot['scores']['groups']
                            if item['code'] == group['code'])
            for team in group['teams']:
                published = next(row for row in official['teams']
                                 if row['team_id'] == team['team_id'])
                self.assertEqual(team['official']['score'], published['score'])
                self.assertEqual(team['official']['rank'], published['rank'])
                self.assertEqual(team['official']['median'], published['median'])

    def test_criteria_are_normalized_to_a_common_scale(self):
        self.seed_biased()
        snapshot = self.publish_after_scores()
        group = snapshot['consistency']['groups'][0]
        for team in group['teams']:
            for card in team['cards']:
                for value in card['normalized_criteria'].values():
                    self.assertGreaterEqual(float(value), 0)
                    self.assertLessEqual(float(value), 100)
                self.assertEqual(card['normalized_total'], card['total'])

    def test_judge_deviation_is_signed_and_paired(self):
        self.seed_biased()
        snapshot = self.publish_after_scores()
        group = snapshot['consistency']['groups'][0]
        high = next(row for row in group['judges'] if row['judge_id'] == 'j0')
        low = next(row for row in group['judges'] if row['judge_id'] == 'j1')
        self.assertEqual(high['reliability']['level'], 'sufficient')
        self.assertGreater(float(high['peer_comparison']['mean_signed_deviation']), 5)
        self.assertLess(float(low['peer_comparison']['mean_signed_deviation']), 0)
        self.assertEqual(high['peer_comparison']['common_teams'], 4)
        self.assertIn('form', high['peer_comparison']['criteria'])

    # -- sample size / partial coverage --------------------------------
    def test_partial_coverage_limits_the_diagnosis(self):
        self.seed_biased(judges=3, teams=4)
        # Drop one judge from one team by inserting a card for a team it did not
        # score in a different way: score only 3 of the 4 teams for j0.
        self.store.inspect('evt')
        with self.store.transaction() as db:
            db.execute('DELETE FROM score_cards WHERE event_id=? AND judge_id=? '
                       'AND team_id=?', ('evt', 'j0', 't3'))
            db.execute('DELETE FROM score_history WHERE event_id=? AND judge_id=? '
                       'AND team_id=?', ('evt', 'j0', 't3'))
        preview = self.publication.preview('evt')
        group = preview['consistency']['groups'][0]
        judge = next(row for row in group['judges'] if row['judge_id'] == 'j0')
        self.assertFalse(judge['coverage_complete'])
        self.assertEqual(judge['coverage'], '0.75')
        self.assertIn('partial_coverage', judge['reliability']['reasons'])
        self.assertEqual(judge['reliability']['level'], 'insufficient')
        self.assertFalse(group['sample']['complete'])
        self.assertIn('incomplete_group',
                      [item['code'] for item in group['diagnostics']])

    def test_small_sample_is_marked_insufficient(self):
        self.competition.create_event('evt', 'Showcase', START, END, 'Asia/Shanghai')
        self.competition.add_group('evt', 'dance', 'Dance', {'form': 60, 'rhythm': 40},
                                   1, 2)
        self.competition.register_team('evt', 't0', 'dance', 'Team 0', 'Qingpu', 3)
        self.competition.assign_judge('evt', 'j0', 'Judge 0', 'dance')
        self.competition.change_state('evt', 'open', self.revision())
        self.sessions.open('evt', 'dance', self.revision())
        self.competition.submit_score('evt', 't0', 'j0',
                                      {'form': 44, 'rhythm': 30}, 's1')
        report = self.publication.preview('evt')['consistency']
        group = report['groups'][0]
        judge = group['judges'][0]
        self.assertEqual(judge['reliability']['level'], 'insufficient')
        self.assertIn('insufficient_teams', judge['reliability']['reasons'])
        self.assertIn('single_judge_panel', judge['reliability']['reasons'])
        self.assertEqual(group['panel_confidence'], 'insufficient')
        self.assertIn('single_judge_panel',
                      [item['code'] for item in group['diagnostics']])

    def test_insufficient_common_teams_blocks_pairwise_diagnosis(self):
        self.seed_biased(judges=3, teams=4)
        with self.store.transaction() as db:
            for team in ('t1', 't2', 't3'):
                db.execute('DELETE FROM score_cards WHERE event_id=? AND judge_id=? '
                           'AND team_id=?', ('evt', 'j1', team))
                db.execute('DELETE FROM score_history WHERE event_id=? AND judge_id=? '
                           'AND team_id=?', ('evt', 'j1', team))
        group = self.publication.preview('evt')['consistency']['groups'][0]
        judge = next(row for row in group['judges'] if row['judge_id'] == 'j1')
        self.assertLess(judge['peer_comparison']['common_teams'], 2)
        self.assertIn('insufficient_common_teams', judge['reliability']['reasons'])
        self.assertEqual(judge['reliability']['level'], 'insufficient')
        pair = next(item for item in group['pairwise']
                    if item['judge_a'] == 'j0' and item['judge_b'] == 'j1')
        self.assertFalse(pair['comparable'])

    # -- revisions -----------------------------------------------------
    def test_revised_card_uses_latest_revision_and_limits_confidence(self):
        self.seed_biased()
        self.competition.submit_score('evt', 't0', 'j0',
                                      {'form': 54, 'rhythm': 36}, 'correction', 1)
        snapshot = self.publish_after_scores()
        group = snapshot['consistency']['groups'][0]
        judge = next(row for row in group['judges'] if row['judge_id'] == 'j0')
        self.assertEqual(judge['reliability']['level'], 'limited')
        self.assertIn('card_revised', judge['reliability']['reasons'])
        self.assertEqual(judge['revisions']['cards_revised'], 1)
        card = next(card for team in group['teams'] for card in team['cards']
                    if team['team_id'] == 't0' and card['judge_id'] == 'j0')
        self.assertEqual(card['revision'], 2)
        self.assertEqual(card['revision_count'], 2)
        self.assertEqual(card['total'], '90.00')

    # -- disputes ------------------------------------------------------
    def test_upheld_dispute_limits_confidence_without_rewriting_card(self):
        self.seed_biased()
        self.disputes.open('evt', 'd1', 't0', 'j0', 'Suspected inflated score')
        self.disputes.resolve('evt', 'd1', 'upheld', 'Panel review noted')
        snapshot = self.publish_after_scores()
        group = snapshot['consistency']['groups'][0]
        judge = next(row for row in group['judges'] if row['judge_id'] == 'j0')
        self.assertEqual(judge['disputes']['upheld'], 1)
        self.assertEqual(judge['reliability']['level'], 'limited')
        self.assertIn('dispute_upheld', judge['reliability']['reasons'])
        official = next(team for team in snapshot['scores']['groups'][0]['teams']
                        if team['team_id'] == 't0')
        card_total = next(card['total'] for team in group['teams']
                          if team['team_id'] == 't0' for card in team['cards']
                          if card['judge_id'] == 'j0')
        self.assertEqual(card_total, '83.00')
        self.assertIsNotNone(official['score'])

    def test_open_dispute_is_reported_as_provisional(self):
        self.seed_biased()
        self.disputes.open('evt', 'd1', 't0', 'j0', 'Open question')
        group = self.publication.preview('evt')['consistency']['groups'][0]
        judge = next(row for row in group['judges'] if row['judge_id'] == 'j0')
        self.assertEqual(judge['reliability']['level'], 'insufficient')
        self.assertIn('open_dispute', judge['reliability']['reasons'])
        self.assertIn('open_dispute',
                      [item['code'] for item in group['diagnostics']])

    # -- frozen versions ----------------------------------------------
    def test_late_backfill_cannot_change_an_older_frozen_report(self):
        self.seed_biased()
        first = self.publish_after_scores()
        first_report = first['consistency']
        # A late service record legitimately creates a new version. It must not
        # silently alter the scoring-consistency report frozen in version 1.
        self.services.record_encounter('evt', 'late', 'tea',
                                       '2026-10-26T10:00:00+08:00', 'visitor', 2)
        second = self.publish()
        self.assertEqual(second['version'], 2)
        # The scoring analysis is unchanged; only its recorded source revision
        # differs because the backfill bumped the event revision.
        self.assertEqual(second['consistency']['groups'], first_report['groups'])
        self.assertEqual(second['consistency']['rules'], first_report['rules'])
        self.assertNotEqual(second['consistency']['source_revision'],
                            first_report['source_revision'])
        self.assertTrue(self.consistency.report('evt', 2)['digest_valid'])
        frozen = self.consistency.report('evt', 1)
        self.assertEqual(frozen['report_digest'], first_report['report_digest'])
        self.assertEqual(frozen['source_revision'], first_report['source_revision'])
        self.assertTrue(frozen['digest_valid'])

    def test_later_dispute_creates_a_new_report_instead_of_rewriting(self):
        self.seed_biased()
        first = self.publish_after_scores()
        first_report = first['consistency']
        self.disputes.open('evt', 'd1', 't0', 'j0', 'Late objection')
        self.disputes.resolve('evt', 'd1', 'upheld', 'Panel agreed')
        second = self.publish()
        self.assertEqual(second['version'], 2)
        self.assertNotEqual(second['consistency']['report_digest'],
                            first_report['report_digest'])
        self.assertEqual(
            self.consistency.report('evt', 1)['report_digest'],
            first_report['report_digest'])
        judge = next(row for row in second['consistency']['groups'][0]['judges']
                     if row['judge_id'] == 'j0')
        self.assertIn('dispute_upheld', judge['reliability']['reasons'])

    def test_report_filtering_by_group_and_judge(self):
        self.seed_biased()
        snapshot = self.publish_after_scores()
        by_group = self.consistency.report('evt', 1, group_code='dance')
        self.assertEqual([group['code'] for group in by_group['groups']], ['dance'])
        by_judge = self.consistency.report('evt', 1, group_code='dance',
                                           judge_id='j0')
        self.assertEqual([row['judge_id'] for row in by_judge['groups'][0]['judges']],
                         ['j0'])
        self.assertTrue(all('j0' in (pair['judge_a'], pair['judge_b'])
                            for pair in by_judge['groups'][0]['pairwise']))
        with self.assertRaises(DomainError) as caught:
            self.consistency.report('evt', 1, judge_id='missing')
        self.assertEqual(caught.exception.code, 'not_found')
        self.assertEqual(snapshot['consistency']['report_digest'],
                         by_group['report_digest'])

    def test_report_requires_a_snapshot(self):
        self.seed()
        with self.assertRaises(DomainError) as caught:
            self.consistency.report('evt')
        self.assertEqual(caught.exception.code, 'snapshot_not_found')

    def test_csv_export_keeps_official_columns(self):
        self.seed_biased()
        self.publish_after_scores()
        report = self.consistency.report('evt', 1, group_code='dance')
        data = self.consistency.csv(report)
        header = data.splitlines()[0]
        self.assertIn('official_rank', header)
        self.assertIn('official_score', header)
        self.assertIn('deviation_from_panel_mean', header)
        self.assertEqual(len([line for line in data.splitlines()[1:]
                              if line]), 12)


class ConsistencyApiTests(Scenario):
    def setUp(self):
        super().setUp()
        self.client = TestClient(create_app(Path(self.temp.name) / 'api.sqlite3'))

    def _publish(self):
        base = '/events/api1'
        self.client.post('/events', json={'id': 'api1', 'name': 'Open day',
                                         'starts_at': START, 'ends_at': END,
                                         'timezone': 'Asia/Shanghai'})
        self.client.post(f'{base}/groups', json={
            'code': 'dance', 'label': 'Dance', 'criteria': {'form': 60, 'rhythm': 40},
            'min_judges': 2, 'max_judges': 3})
        for number in range(3):
            self.client.post(f'{base}/teams', json={
                'id': f't{number}', 'group_code': 'dance', 'name': f'Team {number}',
                'district': 'Qingpu', 'member_count': 2})
        for judge in ('j1', 'j2'):
            self.client.post(f'{base}/judges', json={
                'id': judge, 'name': judge, 'group_code': 'dance'})
        revision = self.client.get(base).json()['revision']
        self.client.post(f'{base}/state', json={'target': 'open',
                                                'expected_revision': revision})
        revision = self.client.get(base).json()['revision']
        self.client.post(f'{base}/groups/dance/session',
                         json={'expected_event_revision': revision})
        for team in range(3):
            for index, judge in enumerate(('j1', 'j2')):
                offset = 5 if judge == 'j1' else 0
                self.client.post(f'{base}/scores', json={
                    'team_id': f't{team}', 'judge_id': judge,
                    'scores': {'form': 40 + offset, 'rhythm': 30 + offset},
                    'request_key': f'{team}-{judge}'})
        revision = self.client.get(base).json()['revision']
        self.client.post(f'{base}/groups/dance/session/close',
                         json={'expected_event_revision': revision})
        revision = self.client.get(base).json()['revision']
        self.client.post(f'{base}/state', json={'target': 'closed',
                                                'expected_revision': revision})
        preview = self.client.get(f'{base}/preview').json()
        return self.client.post(f'{base}/publish', json={
            'expected_head': 0, 'expected_revision': preview['event']['revision']})

    def test_api_report_and_csv(self):
        self.assertEqual(self._publish().status_code, 200)
        response = self.client.get('/events/api1/consistency?version=1&group=dance')
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body['frozen'])
        self.assertTrue(body['digest_valid'])
        self.assertEqual(body['version'], 1)
        judge = next(row for row in body['groups'][0]['judges']
                     if row['judge_id'] == 'j1')
        self.assertGreater(float(judge['peer_comparison']['mean_signed_deviation']), 0)
        exported = self.client.get(
            '/events/api1/export/consistency.csv?version=1&group=dance')
        self.assertEqual(exported.status_code, 200)
        self.assertIn('official_rank', exported.text)

    def test_api_rejects_missing_snapshot_and_unknown_judge(self):
        self.assertEqual(self.client.get('/events/api1/consistency').status_code, 404)
        self._publish()
        self.assertEqual(
            self.client.get('/events/api1/consistency?judge=nobody').status_code, 404)
        self.assertEqual(
            self.client.get('/events/api1/consistency?version=9').status_code, 404)
